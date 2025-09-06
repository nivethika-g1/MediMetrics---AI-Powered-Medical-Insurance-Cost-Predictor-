import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Setup ----------
sns.set(style="whitegrid")
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------- Load Data ----------
CSV_PRIMARY = "medical_insurance_cleaned.csv"
CSV_FALLBACK = "medical_insurance.csv"

if os.path.exists(CSV_PRIMARY):
    csv_path = CSV_PRIMARY
elif os.path.exists(CSV_FALLBACK):
    csv_path = CSV_FALLBACK
else:
    raise FileNotFoundError(
        "CSV not found. Place 'medical_insurance_cleaned.csv' (or 'medical_insurance.csv') in this folder."
    )

df = pd.read_csv(csv_path)

print("\n=== BASIC INFO ===")
print(df.info())
print("\nHead:\n", df.head(), "\n")
print("Shape:", df.shape)
print("Duplicate rows:", df.duplicated().sum())
print("\nDescribe (numeric):\n", df.describe(), "\n")

# ---------- Helper to save plots ----------
def savefig(name):
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, name)
    plt.savefig(out, dpi=120)
    print(f"Saved: {out}")
    plt.show()

# ---------- Univariate Analysis ----------
# Charges
plt.figure(figsize=(8,5))
sns.histplot(df["charges"], kde=True, bins=40)
plt.title("Distribution of Medical Charges")
plt.xlabel("charges"); plt.ylabel("count")
savefig("univariate_charges.png")

# Age
plt.figure(figsize=(8,5))
sns.histplot(df["age"], bins=20)
plt.title("Age Distribution")
plt.xlabel("age"); plt.ylabel("count")
savefig("univariate_age.png")

# BMI
plt.figure(figsize=(8,5))
sns.histplot(df["bmi"], kde=True, bins=30)
plt.title("BMI Distribution")
plt.xlabel("bmi"); plt.ylabel("count")
savefig("univariate_bmi.png")

# Categorical counts
for col in ["sex", "smoker", "region", "children"]:
    plt.figure(figsize=(7,4))
    sns.countplot(x=col, data=df)
    plt.title(f"{col.capitalize()} Count")
    savefig(f"count_{col}.png")

# ---------- Bivariate Analysis ----------
# Charges vs Age (colored by smoker)
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="age", y="charges", hue="smoker", alpha=0.7)
plt.title("Charges vs Age by Smoker")
savefig("bivar_age_charges_by_smoker.png")

# Charges vs BMI (colored by smoker)
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="bmi", y="charges", hue="smoker", alpha=0.7)
plt.title("Charges vs BMI by Smoker")
savefig("bivar_bmi_charges_by_smoker.png")

# Boxplots for charges by categorical vars
for col in ["smoker", "sex", "region", "children"]:
    plt.figure(figsize=(8,5))
    sns.boxplot(data=df, x=col, y="charges")
    plt.title(f"Charges by {col.capitalize()}")
    savefig(f"box_charges_by_{col}.png")

# ---------- Multivariate Views ----------
# Regression view: charges ~ age by smoker
sns.lmplot(
    data=df, x="age", y="charges", hue="smoker", height=5, aspect=1.3, scatter_kws={"alpha":0.5}
)
plt.title("Charges vs Age with Trendlines (by Smoker)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "lm_age_charges_by_smoker.png"), dpi=120)
print("Saved: plots/lm_age_charges_by_smoker.png")
plt.show()

# Regression view: charges ~ bmi by smoker
sns.lmplot(
    data=df, x="bmi", y="charges", hue="smoker", height=5, aspect=1.3, scatter_kws={"alpha":0.5}
)
plt.title("Charges vs BMI with Trendlines (by Smoker)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "lm_bmi_charges_by_smoker.png"), dpi=120)
print("Saved: plots/lm_bmi_charges_by_smoker.png")
plt.show()

# Optional: obesity flag for group insight (BMI > 30)
df["obese"] = (df["bmi"] > 30).astype(int)
plt.figure(figsize=(7,5))
sns.boxplot(data=df, x="obese", y="charges", hue="smoker")
plt.title("Charges by Obesity (BMI>30) and Smoker")
savefig("box_charges_by_obese_smoker.png")

# ---------- Correlation (numeric only) ----------
num_df = df.select_dtypes(include=["float64", "int64"])
corr = num_df.corr()
print("\n=== NUMERIC CORRELATION MATRIX ===\n", corr, "\n")

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap (Numeric Features)")
savefig("corr_heatmap_numeric.png")

# ---------- Outlier Detection ----------
def iqr_outliers(series):
    q1, q3 = np.percentile(series, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    mask = (series < lower) | (series > upper)
    return mask.sum(), lower, upper

for col in ["charges", "bmi"]:
    count, low, high = iqr_outliers(df[col])
    print(f"Outliers in {col}: {count} (IQR bounds: {low:.2f} to {high:.2f})")
    plt.figure(figsize=(8,5))
    sns.boxplot(y=df[col])
    plt.title(f"Outliers in {col}")
    savefig(f"outliers_{col}.png")

# ---------- Quick EDA Summary ----------
print("\n=== QUICK INSIGHTS ===")
print(f"- Rows: {len(df)}; Columns: {df.shape[1]}")
print("- Charges are right-skewed with a long tail (high-cost cases).")
print("- Expect higher charges for smokers and higher BMI, and with increasing age.")
group_smoker = df.groupby("smoker")["charges"].mean().sort_values()
print("\nMean charges by smoker:\n", group_smoker)
group_region = df.groupby("region")["charges"].mean().sort_values()
print("\nMean charges by region:\n", group_region)
group_children = df.groupby("children")["charges"].mean().sort_values()
print("\nMean charges by number of children:\n", group_children)

print("\nAll plots saved under ./plots ✅")
