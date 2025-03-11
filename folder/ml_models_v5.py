# ml_models_v3.py -> REMOVING skew_kurt_product FOR BOTH and rms FOR XGB.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

# Load the SMOTE-balanced dataset
df = pd.read_csv(r"C:\Users\shaur\OneDrive\Desktop\smote_balanced_TD_features_RF_V2.csv")

# Drop "skew_kurt_product" for both models
df = df.drop(columns=["skew_kurt_product"])

# ------------------------ Feature Selection ------------------------
# ✅ Keep "rms" for Random Forest, but remove it for XGBoost
X_rf = df.drop(columns=["label"])  # Keep all for RF
X_xgb = X_rf.drop(columns=["rms"])  # Remove "rms" only for XGB
y = df["label"]

# Encode labels to numerical values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train-Test Split (same 80-20 split for fair comparison)
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    X_rf, y, test_size=0.2, random_state=42, stratify=y
)

X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
    X_xgb, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------ Train Random Forest v3 ------------------------
rf_model_v3 = RandomForestClassifier(n_estimators=500, max_features="sqrt", random_state=42)
rf_model_v3.fit(X_train_rf, y_train_rf)

# Predict and evaluate RF v3
y_pred_rf_v3 = rf_model_v3.predict(X_test_rf)
accuracy_rf_v3 = accuracy_score(y_test_rf, y_pred_rf_v3)
print(f"\n🔹 Random Forest v3 Accuracy: {accuracy_rf_v3:.4f}")
print(classification_report(y_test_rf, y_pred_rf_v3, target_names=label_encoder.classes_))

# ------------------------ Train XGBoost v3 ------------------------
xgb_model_v3 = xgb.XGBClassifier(objective="multi:softmax", num_class=len(set(y_train_xgb)),
                                  max_depth=6, eta=0.3, n_estimators=100, random_state=42)
xgb_model_v3.fit(X_train_xgb, y_train_xgb)

# Predict and evaluate XGB v3
y_pred_xgb_v3 = xgb_model_v3.predict(X_test_xgb)
accuracy_xgb_v3 = accuracy_score(y_test_xgb, y_pred_xgb_v3)
print(f"\n🔹 XGBoost v3 Accuracy: {accuracy_xgb_v3:.4f}")
print(classification_report(y_test_xgb, y_pred_xgb_v3, target_names=label_encoder.classes_))

print("\n✅ Model training completed. Ready for validation & hyperparameter tuning!")
