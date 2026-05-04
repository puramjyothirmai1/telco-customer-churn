"""
Generate figures for the Worthwhile assessment report.
Runs the full churn analysis on the Telco dataset and saves PNGs to ./figures.
"""
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
)

SEED = 42
np.random.seed(SEED)
sns.set_theme(style="whitegrid")
os.makedirs("figures", exist_ok=True)

DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
service_cols = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "MultipleLines",
]
for c in service_cols:
    df[c] = df[c].replace({"No internet service": "No", "No phone service": "No"})

df["Churn"] = (df["Churn"] == "Yes").astype(int)
df = df.drop(columns=["customerID"])

y = df["Churn"]
X = df.drop(columns=["Churn"])
num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
cat_cols = [c for c in X.columns if c not in num_cols]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

print("Class balance:")
print(y.value_counts(normalize=True).round(3))


def save(fig_or_none, name):
    plt.tight_layout()
    plt.savefig(f"figures/{name}", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"saved figures/{name}")


plt.figure(figsize=(5, 4))
ax = sns.countplot(x=y.map({0: "No", 1: "Yes"}), palette="Blues")
total = len(y)
for p in ax.patches:
    pct = 100 * p.get_height() / total
    ax.annotate(f"{int(p.get_height())} ({pct:.1f}%)",
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom")
plt.title("Churn distribution")
plt.xlabel("Churn")
plt.ylabel("Customers")
save(None, "01_churn_distribution.png")

plt.figure(figsize=(6, 4))
sns.countplot(
    data=df, x="Contract",
    hue=df["Churn"].map({0: "No", 1: "Yes"}),
    palette="Blues",
)
plt.title("Churn by contract type")
plt.ylabel("Customers")
plt.legend(title="Churn")
save(None, "02_churn_by_contract.png")

plt.figure(figsize=(7, 4))
sns.histplot(
    data=df, x="tenure",
    hue=df["Churn"].map({0: "No", 1: "Yes"}),
    multiple="stack", bins=30, palette="Blues",
)
plt.title("Churn by tenure (months)")
plt.xlabel("Tenure (months)")
save(None, "03_churn_by_tenure.png")

plt.figure(figsize=(7, 4))
sns.kdeplot(
    data=df, x="MonthlyCharges",
    hue=df["Churn"].map({0: "No", 1: "Yes"}),
    common_norm=False, fill=True, palette="Blues",
)
plt.title("Monthly charges by churn")
plt.xlabel("Monthly charges (USD)")
save(None, "04_monthly_charges_by_churn.png")

plt.figure(figsize=(6, 5))
sns.heatmap(
    df[num_cols + ["Churn"]].corr(),
    annot=True, cmap="Blues", fmt=".2f",
)
plt.title("Correlation heatmap (numeric features)")
save(None, "05_correlation_heatmap.png")

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

logreg = Pipeline([
    ("pre", preprocess),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
])
gb = Pipeline([
    ("pre", preprocess),
    ("clf", GradientBoostingClassifier(random_state=SEED)),
])

models = {"Logistic Regression": logreg, "Gradient Boosting": gb}
results = {}
metrics_rows = []
for name, model in models.items():
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, y_proba)
    print(f"\n=== {name} ===")
    rep = classification_report(y_te, y_pred, digits=3, output_dict=True)
    print(classification_report(y_te, y_pred, digits=3))
    print("ROC-AUC:", round(auc, 3))
    print("Confusion matrix:\n", confusion_matrix(y_te, y_pred))
    results[name] = {"model": model, "auc": auc, "y_pred": y_pred, "y_proba": y_proba}
    metrics_rows.append({
        "Model": name,
        "Accuracy": round(rep["accuracy"], 3),
        "Precision (churn)": round(rep["1"]["precision"], 3),
        "Recall (churn)": round(rep["1"]["recall"], 3),
        "F1 (churn)": round(rep["1"]["f1-score"], 3),
        "ROC-AUC": round(auc, 3),
    })

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv("figures/metrics_summary.csv", index=False)
print("\nMetrics summary:\n", metrics_df)

best_name = max(results, key=lambda k: results[k]["auc"])
best = results[best_name]

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_te, best["y_pred"],
    display_labels=["No churn", "Churn"], cmap="Blues", ax=ax,
)
plt.title(f"Confusion matrix - {best_name}")
save(None, "06_confusion_matrix.png")

fig, ax = plt.subplots(figsize=(6, 5))
for name, r in results.items():
    RocCurveDisplay.from_predictions(y_te, r["y_proba"], name=name, ax=ax)
plt.title("ROC curves")
save(None, "07_roc_curves.png")

gb_model = results["Gradient Boosting"]["model"]
ohe = gb_model.named_steps["pre"].named_transformers_["cat"]
feature_names = num_cols + list(ohe.get_feature_names_out(cat_cols))
importances = gb_model.named_steps["clf"].feature_importances_
fi = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15)

plt.figure(figsize=(7, 6))
fi[::-1].plot(kind="barh", color="#1f77b4")
plt.title("Top 15 features - Gradient Boosting")
plt.xlabel("Importance")
save(None, "08_feature_importance.png")

print("\nDone. Figures in ./figures")
