# MediMetrics — AI Powered Medical Insurance Cost Predictor

An end-to-end **machine learning project** that predicts individual **medical insurance charges** from demographic and health data, complete with **EDA**, **multiple regression models**, **MLflow tracking**, and an **interactive Streamlit app**.

---

## 📌 Overview
This project takes you from **raw medical insurance data** to a **fully deployed prediction app**.  
It includes:
- Data cleaning & preprocessing
- Exploratory Data Analysis (EDA)
- Training 5 regression models
- MLflow experiment tracking
- Streamlit web app for cost prediction
- Confidence intervals for predictions
- Downloadable prediction receipts (CSV & HTML)

---

## 🗂 Dataset
**Columns:**
- `age` — Age of the insured
- `sex` — Gender of the insured
- `bmi` — Body Mass Index
- `children` — Number of children/dependents
- `smoker` — Smoking status (`yes` / `no`)
- `region` — Residential region in the US
- `charges` — Individual medical costs billed by health insurance

📌 **Target:** `charges` (continuous numerical variable)

---

## 📊 EDA Highlights
- Charges are **right-skewed** — few people have very high medical costs.
- Smokers have **significantly higher** charges.
- BMI > 30 (obese) shows a sharp cost increase, especially for smokers.
- Age has a **positive correlation** with charges.
- Regions and number of children show smaller, but noticeable, effects.

EDA plots are available in the **`plots/`** folder and inside the app’s **EDA Insights** tab.

---

## ⚙️ Tech Stack
- **Python**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **ML Models**: Scikit-learn, XGBoost
- **Experiment Tracking**: MLflow
- **Deployment**: Streamlit
- **Model Persistence**: Joblib

---

## 📈 Model Performance

| Model              | RMSE    | MAE    | R²    |
|--------------------|---------|--------|-------|
| Linear Regression  | 5956.34 | 4177.05| 0.8069|
| Ridge              | 5963.75 | 4185.52| 0.8064|
| Lasso              | 5956.34 | 4177.05| 0.8069|
| **Random Forest**  | **4608.87** | **2537.13** | **0.8844** |
| XGBoost            | 4772.17 | 2711.14| 0.8761|

**Best Model:** Random Forest  
**Validation RMSE:** ~4608.87 (used to compute 95% confidence intervals in the app)

---

## 📦 Project Structure
```plaintext
Medical Insurance Cost Prediction/
├─ app.py                 # Streamlit app
├─ eda.py                 # EDA & plots
├─ train.py               # Model training + MLflow logging
├─ medical_insurance_cleaned.csv
├─ plots/                 # Generated EDA visualizations
├─ best_model.pkl         # Trained best model
├─ best_metrics.json      # Best model info + RMSE
├─ results.csv            # Metrics for all models
├─ requirements.txt
└─ README.md
