import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from preprocessing import preprocess_data
from feature_engineering import add_features


# ===============================
# 1️⃣ Load Dataset
# ===============================
df = pd.read_csv("data/European_Bank.csv")

# ===============================
# 2️⃣ Preprocessing
# ===============================
df = preprocess_data(df)
df = add_features(df)

# ===============================
# 3️⃣ Split Features & Target
# ===============================
X = df.drop("Exited", axis=1)
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ===============================
# 4️⃣ Train Model
# ===============================
model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# ===============================
# 5️⃣ Predictions
# ===============================
y_prob = model.predict_proba(X_test)[:, 1]

# 🔥 Custom Threshold (Tune This)
threshold = 0.35

y_pred_custom = (y_prob >= threshold).astype(int)

# Risk Scoring Bands
def assign_risk(prob):
    if prob < 0.30:
        return "Low Risk"
    elif prob < 0.70:
        return "Medium Risk"
    else:
        return "High Risk"

risk_levels = [assign_risk(p) for p in y_prob]

risk_df = pd.DataFrame({
    "Actual": y_test.values,
    "Probability": y_prob,
    "Risk_Level": risk_levels
})

print("\nSample Risk Predictions:\n")
print(risk_df.head())

# ===============================
# 6️⃣ Evaluation
# ===============================
print("\n===== MODEL PERFORMANCE =====\n")

print("Threshold Used:", threshold)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_custom))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred_custom))

print("\nROC-AUC Score:", roc_auc_score(y_test, y_prob))

# ===============================
# 7️⃣ Save Model
# ===============================
joblib.dump(model, "models/churn_model.pkl")

print("\nModel saved successfully in models/churn_model.pkl")

# ===============================
# 8️⃣ Feature Importance
# ===============================

import matplotlib.pyplot as plt

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features:\n")
print(feature_importance.head(10))

# Plot
plt.figure(figsize=(8,6))
plt.barh(feature_importance["Feature"][:10],
         feature_importance["Importance"][:10])
plt.gca().invert_yaxis()
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.savefig("models/shap_summary.png")
plt.close()

# ===============================
# 9️⃣ SHAP Explainability (Final Clean Version)
# ===============================

import shap
import matplotlib.pyplot as plt

print("\nGenerating SHAP summary plot...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Use only class 1 (churn class)
if isinstance(shap_values, list):
    shap_vals = shap_values[1]
else:
    shap_vals = shap_values

# Create clean bar summary plot
plt.figure(figsize=(10,6))
shap.summary_plot(shap_vals, X_test, plot_type="bar", show=False)

plt.tight_layout()
plt.savefig("models/shap_summary.png")
plt.close()

print("SHAP summary plot saved in models/shap_summary.png")
# Save feature order
joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")