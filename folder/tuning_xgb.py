import pandas as pd
import numpy as np
import time
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

# Load the SMOTE-balanced dataset
df = pd.read_csv(r"C:\Users\shaur\OneDrive\Desktop\smote_balanced_TD_features_RF_V2.csv")

# Apply final feature selection
df_rf = df.copy()  # ✅ Keep all features for RF
df_xgb = df.drop(columns=["rms", "skew_kurt_ratio"])  # ✅ Remove "rms" and "skew_kurt_ratio" for XGBoost

# Separate features and labels
X_rf = df_rf.drop(columns=["label"])
X_xgb = df_xgb.drop(columns=["label"])
y = df["label"]

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train-Test Split
X_rf_train, X_rf_test, y_train_rf, y_test_rf = train_test_split(
    X_rf, y, test_size=0.2, random_state=42, stratify=y
)
X_xgb_train, X_xgb_test, y_train_xgb, y_test_xgb = train_test_split(
    X_xgb, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------ Hyperparameter Tuning for XGBoost ------------------------
xgb_param_dist = {
    "n_estimators": [100, 300, 500, 700],
    "max_depth": [3, 6, 9, 12],
    "learning_rate": [0.01, 0.1, 0.2, 0.3, 0.5],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 0.1, 0.2, 0.5]
}

print("\n🔹 Tuning XGBoost (Using RandomizedSearchCV)...")
start_time = time.time()

xgb_random_search = RandomizedSearchCV(
    xgb.XGBClassifier(objective="multi:softmax", num_class=len(set(y_train_xgb))),
    param_distributions=xgb_param_dist,
    n_iter=30,
    cv=5,
    n_jobs=-1,
    verbose=1,
    random_state=42
)
xgb_random_search.fit(X_xgb_train, y_train_xgb)

# Get best model
xgb_best = xgb_random_search.best_estimator_

print(f"\n✅ Best XGB Parameters: {xgb_random_search.best_params_}")
print(f"⏳ Time Taken: {time.time() - start_time:.2f} seconds")

# Evaluate the best XGB model
y_pred_xgb_best = xgb_best.predict(X_xgb_test)
accuracy_xgb_best = accuracy_score(y_test_xgb, y_pred_xgb_best)
print(f"\n🔹 Tuned XGBoost Accuracy: {accuracy_xgb_best:.4f}")
print(classification_report(y_test_xgb, y_pred_xgb_best, target_names=label_encoder.classes_))

# ------------------------ ✅ Save the XGBoost Model ------------------------
save_path = r"C:\Users\shaur\OneDrive\Desktop\xgb_tuned_model.pkl"

# Save the model
joblib.dump(xgb_best, save_path)

print(f"\n✅ XGBoost model saved successfully at: {save_path}")

print("\n✅ Hyperparameter tuning complete! 🚀 Model is saved and ready for validation.")
