import json
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from mlflow.models.signature import infer_signature

# ---------- Load Data ----------
df = pd.read_csv("medical_insurance_cleaned.csv")
X = df.drop("charges", axis=1)
y = df["charges"]

categorical_cols = ["sex", "smoker", "region"]
numeric_cols = ["age", "bmi", "children"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.001),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("Medical Insurance Cost Prediction")

results = []
best_model_name, best_rmse, best_pipeline = None, float("inf"), None

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # MLflow: log params/metrics + model (with signature & input example)
        mlflow.log_param("model_name", name)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)

        input_example = X_train.iloc[[0]].copy()
        signature = infer_signature(X_train.head(5), pipeline.predict(X_train.head(5)))
        mlflow.sklearn.log_model(
            pipeline,
            name,
            signature=signature,
            input_example=input_example,
        )

        results.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2})

        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_pipeline = pipeline

# Save artifacts locally for the app
joblib.dump(best_pipeline, "best_model.pkl")
pd.DataFrame(results).to_csv("results.csv", index=False)
with open("best_metrics.json", "w") as f:
    json.dump(
        {"best_model": best_model_name, "RMSE": float(best_rmse),
         "note": "RMSE computed on held-out test set (20%)."}, f, indent=2
    )

print(f"Best model: {best_model_name} with RMSE = {best_rmse:.2f}")
print(pd.DataFrame(results))
