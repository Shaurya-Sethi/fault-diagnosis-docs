# rf_robustness_checks.py
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the SMOTE-balanced dataset
df = pd.read_csv(r"C:\Users\shaur\OneDrive\Desktop\smote_balanced_TD_features_RF_V2.csv")
df = df.drop(columns=["skew_kurt_product"])

# Final Feature Set (Keeping "rms" for RF)
X_rf = df.drop(columns=["label"])
y = df["label"]

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_rf, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize RF with best hyperparameters
rf_best = RandomForestClassifier(
    n_estimators=500, max_features="sqrt",
    min_samples_leaf=2, min_samples_split=2,
    max_depth=None, random_state=42
)

# ------------------------ 1️⃣ Stability Across Multiple Train-Test Splits ------------------------
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_accuracies = []

print("\n🔹 Checking Random Forest Stability Across Splits...")
for train_idx, test_idx in kf.split(X_rf, y):
    X_train_k, X_test_k = X_rf.iloc[train_idx], X_rf.iloc[test_idx]
    y_train_k, y_test_k = y[train_idx], y[test_idx]

    rf_best.fit(X_train_k, y_train_k)
    accuracy = accuracy_score(y_test_k, rf_best.predict(X_test_k))
    rf_accuracies.append(accuracy)

print(f"\n✅ Stability Check: Mean Accuracy = {np.mean(rf_accuracies):.4f}, Std Dev = {np.std(rf_accuracies):.4f}")

# ------------------------ 2️⃣ Cross-Validation Performance ------------------------
print("\n🔹 Running 5-Fold Cross-Validation...")
cv_scores = cross_val_score(rf_best, X_rf, y, cv=5)
print(f"\n✅ Cross-Validation: Mean Accuracy = {np.mean(cv_scores):.4f}, Std Dev = {np.std(cv_scores):.4f}")

# ------------------------ 3️⃣ Confusion Matrix Analysis ------------------------
print("\n🔹 Generating Confusion Matrix...")
rf_best.fit(X_train, y_train)
y_pred_rf_best = rf_best.predict(X_test)

cm = confusion_matrix(y_test, y_pred_rf_best)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.show()

# Print classification report
print("\n🔹 Classification Report (RF Best):")
print(classification_report(y_test, y_pred_rf_best, target_names=label_encoder.classes_))

# ------------------------ 4️⃣ Model Interpretability Using SHAP ------------------------

print("\n🔹 Running SHAP for Feature Interpretability (Averaging over Classes)...")
import shap
import numpy as np

# Convert X_test to a NumPy array (if it's not already)
X_test_array = X_test.to_numpy()

# Initialize TreeExplainer for the RF model
explainer = shap.TreeExplainer(rf_best)
# Get SHAP values using the TreeExplainer.
# For multiclass models, shap_values is typically a list (or array) with shape (n_samples, n_features, n_classes)
shap_values = explainer.shap_values(X_test_array)

# If shap_values is a list, convert it to an array with shape (n_samples, n_features, n_classes)
if isinstance(shap_values, list):
    shap_values = np.stack(shap_values, axis=-1)  # Now shape should be (n_samples, n_features, n_classes)

print("SHAP Values Shape:", shap_values.shape)

# Average the absolute SHAP values over the class dimension to get a (n_samples, n_features) array
shap_values_mean = np.mean(np.abs(shap_values), axis=2)

# Plot the SHAP summary plot
shap.summary_plot(shap_values_mean, X_test_array, feature_names=X_rf.columns)

