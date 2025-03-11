import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------ ✅ Load the Saved XGBoost Model ------------------------
xgb_model_path = r"C:\Users\shaur\OneDrive\Desktop\xgb_tuned_model.pkl"
xgb_best = joblib.load(xgb_model_path)

print(f"\n✅ Loaded XGBoost Model from: {xgb_model_path}")

# ------------------------ ✅ Load Dataset & Preprocess ------------------------
df = pd.read_csv("C:/Users/shaur/OneDrive/Desktop/smote_balanced_TD_features_RF_V2.csv")

# Apply final feature selection (Removing "rms" and "skew_kurt_ratio" for XGB)
df_xgb = df.drop(columns=["rms", "skew_kurt_ratio"])

# Separate features and labels
X_xgb = df_xgb.drop(columns=["label"])
y = df["label"]

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_xgb, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------ 1️⃣ Stability Across Multiple Train-Test Splits ------------------------
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
xgb_accuracies = []

print("\n🔹 Checking XGBoost Stability Across Splits...")
for train_idx, test_idx in kf.split(X_xgb, y):
    X_train_k, X_test_k = X_xgb.iloc[train_idx], X_xgb.iloc[test_idx]
    y_train_k, y_test_k = y[train_idx], y[test_idx]

    xgb_best.fit(X_train_k, y_train_k)
    accuracy = accuracy_score(y_test_k, xgb_best.predict(X_test_k))
    xgb_accuracies.append(accuracy)

print(f"\n✅ Stability Check: Mean Accuracy = {np.mean(xgb_accuracies):.4f}, Std Dev = {np.std(xgb_accuracies):.4f}")

# ------------------------ 2️⃣ Cross-Validation Performance ------------------------
print("\n🔹 Running 5-Fold Cross-Validation...")
cv_scores = cross_val_score(xgb_best, X_xgb, y, cv=5)
print(f"\n✅ Cross-Validation: Mean Accuracy = {np.mean(cv_scores):.4f}, Std Dev = {np.std(cv_scores):.4f}")

# ------------------------ 3️⃣ Confusion Matrix Analysis ------------------------
print("\n🔹 Generating Confusion Matrix...")
y_pred_xgb_best = xgb_best.predict(X_test)

cm = confusion_matrix(y_test, y_pred_xgb_best)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - XGBoost")
plt.show()

# Print classification report
print("\n🔹 Classification Report (XGB Best):")
print(classification_report(y_test, y_pred_xgb_best, target_names=label_encoder.classes_))

# ------------------------ 4️⃣ Model Interpretability Using SHAP ------------------------
print("\n🔹 Running SHAP for Feature Interpretability (Using TreeExplainer)...")

# Convert X_test to a NumPy array (if it's not already)
X_test_array = X_test.to_numpy()

# Initialize TreeExplainer for XGBoost
explainer = shap.TreeExplainer(xgb_best)
shap_values = explainer.shap_values(X_test_array)

# Check SHAP values shape for multiclass handling
if isinstance(shap_values, list):
    shap_values = np.stack(shap_values, axis=-1)  # Convert to shape (n_samples, n_features, n_classes)

print("SHAP Values Shape:", shap_values.shape)

# Average SHAP values across all classes
shap_values_mean = np.mean(np.abs(shap_values), axis=2)

# SHAP Summary Plot
shap.summary_plot(shap_values_mean, X_test_array, feature_names=X_xgb.columns)
