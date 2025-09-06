# app.py — Medical Insurance Cost Prediction (Tabs + Form + Downloadable Receipt + 95% Range)
import os, sys, subprocess, io, json, datetime as dt
import pandas as pd
import joblib

# ----- Dependencies -----
for pkg in ("streamlit", "joblib", "pandas"):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
import streamlit as st  # after potential install

st.set_page_config(page_title="Medical Insurance Cost Prediction", page_icon="💊", layout="wide")

# ----- Load model & metrics -----
@st.cache_resource(show_spinner=False)
def load_model(path="best_model.pkl"):
    if not os.path.exists(path):
        st.error("❌ best_model.pkl not found. Please run train.py first.")
        st.stop()
    return joblib.load(path)

@st.cache_data(show_spinner=False)
def load_metrics(path="best_metrics.json"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

model = load_model()
best_metrics = load_metrics()  # may be None

# ----- Helpers -----
def predict_cost(model, age, sex, bmi, children, smoker, region) -> float:
    X = pd.DataFrame({
        "age": [age], "sex": [sex], "bmi": [bmi],
        "children": [children], "smoker": [smoker], "region": [region],
    })
    return float(model.predict(X)[0])

@st.cache_data(show_spinner=False)
def list_plot_images(folder="plots"):
    if not os.path.isdir(folder): return []
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".svg"))]

def build_receipt_df(age, sex, bmi, children, smoker, region, pred, ts):
    return pd.DataFrame([{
        "timestamp": ts.isoformat(sep=" ", timespec="seconds"),
        "age": age, "sex": sex, "bmi": bmi,
        "children": children, "smoker": smoker, "region": region,
        "predicted_charges": round(pred, 2)
    }])

def build_receipt_html(df: pd.DataFrame) -> str:
    r = df.iloc[0].to_dict()
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Medical Insurance Cost Prediction — Receipt</title>
<style>
body{{font-family:Arial,sans-serif;margin:24px}}h1{{margin-bottom:4px}}.sub{{color:#666;margin-top:0}}
table{{border-collapse:collapse;width:100%;margin-top:16px}}th,td{{border:1px solid #ddd;padding:10px;text-align:left}}
th{{background:#f7f7f7}}.amount{{font-weight:bold;font-size:1.2rem}}.note{{margin-top:16px;color:#555}}
</style></head><body>
<h1>Medical Insurance Cost Prediction — Receipt</h1>
<p class="sub">Generated: {r["timestamp"]}</p>
<table>
<tr><th>Age</th><td>{r["age"]}</td></tr>
<tr><th>Sex</th><td>{r["sex"]}</td></tr>
<tr><th>BMI</th><td>{r["bmi"]}</td></tr>
<tr><th>Children</th><td>{r["children"]}</td></tr>
<tr><th>Smoker</th><td>{r["smoker"]}</td></tr>
<tr><th>Region</th><td>{r["region"]}</td></tr>
<tr><th>Predicted Charges</th><td class="amount">${r["predicted_charges"]:,.2f}</td></tr>
</table>
<p class="note">Note: This is a model estimate based on historical data and features provided. It is not medical advice.</p>
</body></html>"""

# ----- UI -----
st.title("💊 Medical Insurance Cost Prediction")
st.caption("Estimate medical insurance charges based on demographic and health details. Use the EDA tab for insights.")

tab_pred, tab_eda = st.tabs(["🔮 Prediction", "📊 EDA Insights"])

# ========= PREDICTION =========
with tab_pred:
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("Enter Your Details")
        with st.form("prediction_form", clear_on_submit=False):
            age = st.slider("Age", 18, 100, 30, key="age_slider")
            sex = st.selectbox("Sex", ["male", "female"], key="sex_select")
            bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0, step=0.1, key="bmi_input")
            children = st.selectbox("Number of Children", [0, 1, 2, 3, 4, 5], key="children_select")
            smoker = st.selectbox("Smoker", ["yes", "no"], key="smoker_select")
            region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"], key="region_select")
            submitted = st.form_submit_button("Predict Insurance Cost", use_container_width=True)

        if submitted:
            pred = predict_cost(model, age, sex, bmi, children, smoker, region)
            st.success(f"💵 Estimated Insurance Cost: **${pred:,.2f}**")

            # Optional 95% prediction interval using test RMSE
            if best_metrics and "RMSE" in best_metrics:
                rmse = float(best_metrics["RMSE"])
                delta = 1.96 * rmse  # ~95% interval
                low = max(0.0, pred - delta)
                high = max(0.0, pred + delta)
                st.caption(f"Approx. 95% range based on validation RMSE ({rmse:,.0f}): **${low:,.0f} – ${high:,.0f}**")

            ts = dt.datetime.now()
            receipt_df = build_receipt_df(age, sex, bmi, children, smoker, region, pred, ts)

            # Download CSV
            csv_buf = io.StringIO()
            receipt_df.to_csv(csv_buf, index=False)
            st.download_button(
                "⬇️ Download CSV Receipt",
                data=csv_buf.getvalue(),
                file_name=f"insurance_prediction_{ts.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            # Download printable HTML
            html_content = build_receipt_html(receipt_df)
            st.download_button(
                "⬇️ Download Printable HTML Receipt",
                data=html_content.encode("utf-8"),
                file_name=f"insurance_prediction_{ts.strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                use_container_width=True
            )

            with st.expander("See input as model-ready row"):
                st.dataframe(receipt_df, use_container_width=True)

    with right:
        st.subheader("How to interpret the prediction")
        st.markdown(
            """
            - **Higher age / BMI** generally increase charges.  
            - **Smoker = yes** has a strong positive effect on charges.  
            - Regions and number of children have smaller but non-zero impact.  
            - This model is a **Random Forest** trained on your dataset; it doesn’t diagnose conditions.  
            """
        )
        if best_metrics:
            st.info(f"Best model: **{best_metrics.get('best_model','N/A')}** · Test RMSE: **{best_metrics.get('RMSE', 'N/A')}**")

# ========= EDA =========
with tab_eda:
    st.subheader("Exploratory Data Analysis")
    st.caption("These plots were generated by your `eda.py` and saved to the `plots/` folder.")
    imgs = list_plot_images("plots")
    if not imgs:
        st.info("No images found in the `plots/` folder. Run `python eda.py` to generate EDA visuals.")
    else:
        for i in range(0, len(imgs), 2):
            cols = st.columns(2)
            for c, p in zip(cols, imgs[i:i+2]):
                c.image(p, caption=os.path.basename(p), use_container_width=True)
