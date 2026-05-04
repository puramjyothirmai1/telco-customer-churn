"""
Customer Churn Prediction - Telco Customer Churn (Public Dataset)
Author: Jyothirmai Puram
Submission: Worthwhile Remote Data Scientist Assessment

Dataset: Telco Customer Churn (Kaggle / IBM Sample Data)
URL:     https://www.kaggle.com/datasets/blastchar/telco-customer-churn
File:    WA_Fn-UseC_-Telco-Customer-Churn.csv

This script is structured as a Jupyter-style notebook flow.
Each `# %%` block corresponds to a notebook cell and can be run in
VS Code, PyCharm, or any IDE with Jupyter cell support.

Run:
    pip install pandas numpy matplotlib seaborn scikit-learn
    python churn_analysis.py
"""

# %% [markdown]
# # 1. Imports and configuration

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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

DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"


# %% [markdown]
# # 2. Load data

# %%
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print(df.head())


# %% [markdown]
# # 3. Basic checks

# %%
print(df.dtypes)
print("Duplicates on customerID:", df.duplicated(subset=["customerID"]).sum())
print("Class balance:")
print(df["Churn"].value_counts(normalize=True).round(3))


# %% [markdown]
# # 4. Cleaning
# - TotalCharges has blank strings for tenure=0 customers - coerce and fill with 0.
# - Collapse "No internet service" and "No phone service" to "No" so the
#   information is not encoded twice.
# - Encode the target.
# - Drop the ID column.

# %%
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

service_cols = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "MultipleLines",
]
for c in service_cols:
    df[c] = df[c].replace(
        {"No internet service": "No", "No phone service": "No"}
    )

df["Churn"] = (df["Churn"] == "Yes").astype(int)
df = df.drop(columns=["customerID"])


# %% [markdown]
# # 5. Train / test split (stratified)

# %%
y = df["Churn"]
X = df.drop(columns=["Churn"])

num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
cat_cols = [c for c in X.columns if c not in num_cols]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)


# %% [markdown]
# # 6. Visualizations

# %%
sns.countplot(x=y.map({0: "No", 1: "Yes"}))
plt.title("Churn distribution")
plt.show()

sns.countplot(
    data=df, x="Contract", hue=df["Churn"].map({0: "No", 1: "Yes"})
)
plt.title("Churn by contract type")
plt.show()

sns.histplot(
    data=df,
    x="tenure",
    hue=df["Churn"].map({0: "No", 1: "Yes"}),
    multiple="stack",
    bins=30,
)
plt.title("Churn by tenure (months)")
plt.show()

sns.kdeplot(
    data=df,
    x="MonthlyCharges",
    hue=df["Churn"].map({0: "No", 1: "Yes"}),
    common_norm=False,
    fill=True,
)
plt.title("Monthly charges by churn")
plt.show()

plt.figure(figsize=(6, 4))
sns.heatmap(df[num_cols + ["Churn"]].corr(), annot=True, cmap="Blues")
plt.title("Correlation heatmap (numeric features)")
plt.show()


# %% [markdown]
# # 7. Preprocessing pipeline

# %%
preprocess = ColumnTransformer(
    [
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)


# %% [markdown]
# # 8. Models
# - Logistic Regression as a baseline (simple, explainable).
# - Gradient Boosting as a stronger model (handles non-linear interactions).

# %%
logreg = Pipeline(
    [
        ("pre", preprocess),
        (
            "clf",
            LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=SEED
            ),
        ),
    ]
)

gb = Pipeline(
    [
        ("pre", preprocess),
        ("clf", GradientBoostingClassifier(random_state=SEED)),
    ]
)

models = {"Logistic Regression": logreg, "Gradient Boosting": gb}


# %% [markdown]
# # 9. Fit and evaluate

# %%
results = {}
for name, model in models.items():
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, y_proba)

    print(f"=== {name} ===")
    print(classification_report(y_te, y_pred, digits=3))
    print(f"ROC-AUC: {auc:.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_te, y_pred))
    print()

    results[name] = {
        "model": model,
        "auc": auc,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


# %% [markdown]
# # 10. Confusion matrix and ROC curves for the better model

# %%
best_name = max(results, key=lambda k: results[k]["auc"])
best = results[best_name]

ConfusionMatrixDisplay.from_predictions(y_te, best["y_pred"])
plt.title(f"Confusion matrix - {best_name}")
plt.show()

fig, ax = plt.subplots()
for name, r in results.items():
    RocCurveDisplay.from_predictions(y_te, r["y_proba"], name=name, ax=ax)
plt.title("ROC curves")
plt.show()


# %% [markdown]
# # 11. Feature importance (Gradient Boosting)

# %%
gb_model = results["Gradient Boosting"]["model"]
ohe = gb_model.named_steps["pre"].named_transformers_["cat"]
feature_names = num_cols + list(ohe.get_feature_names_out(cat_cols))
importances = gb_model.named_steps["clf"].feature_importances_

fi = (
    pd.Series(importances, index=feature_names)
    .sort_values(ascending=False)
    .head(15)
)

plt.figure(figsize=(7, 5))
fi[::-1].plot(kind="barh")
plt.title("Top 15 features - Gradient Boosting")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()


# %% [markdown]
# # 12. Optional - Random Forest hyperparameter tuning sketch
# Uncomment the `search.fit(...)` line to run; it can take a few minutes.

# %%
rf = Pipeline(
    [
        ("pre", preprocess),
        (
            "clf",
            RandomForestClassifier(
                class_weight="balanced", random_state=SEED, n_jobs=-1
            ),
        ),
    ]
)

param_dist = {
    "clf__n_estimators": [200, 400, 600, 800],
    "clf__max_depth": [None, 6, 10, 16, 24],
    "clf__min_samples_split": [2, 5, 10],
    "clf__min_samples_leaf": [1, 2, 4],
    "clf__max_features": ["sqrt", "log2", 0.5],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=20,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    random_state=SEED,
    verbose=1,
)
# search.fit(X_tr, y_tr)
# print("Best params:", search.best_params_)
# print("Best CV ROC-AUC:", round(search.best_score_, 3))
