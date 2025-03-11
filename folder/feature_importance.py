# feature_importance.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load the SMOTE-balanced dataset
df = pd.read_csv(r"C:\Users\shaur\OneDrive\Desktop\smote_balanced_TD_features_RF_V2.csv")

# Separate features and labels
X = df.drop(columns=["label"])  # Features
y = df["label"]

# Encode labels to numerical values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train-Test Split (same 80-20 split for consistency)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------ Feature Importance with Random Forest ------------------------
rf_model = RandomForestClassifier(n_estimators=500, max_features="sqrt", random_state=42)
rf_model.fit(X_train, y_train)

# Get feature importances from RF
rf_importance = rf_model.feature_importances_

# ------------------------ Feature Importance with XGBoost ------------------------
xgb_model = xgb.XGBClassifier(objective="multi:softmax", num_class=len(set(y_train)),
                              max_depth=6, eta=0.3, n_estimators=100)
xgb_model.fit(X_train, y_train)

# Get feature importances from XGBoost
xgb_importance = xgb_model.feature_importances_

# ------------------------ Visualizing Feature Importance ------------------------
# Convert feature importance to DataFrame
feature_names = X.columns
rf_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": rf_importance})
xgb_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": xgb_importance})

# Sort features by importance
rf_importance_df = rf_importance_df.sort_values(by="Importance", ascending=False)
xgb_importance_df = xgb_importance_df.sort_values(by="Importance", ascending=False)

# Plot Feature Importance for RF
plt.figure(figsize=(10, 5))
sns.barplot(x="Importance", y="Feature", data=rf_importance_df, palette="Blues_r")
plt.title("Feature Importance (Random Forest)")
plt.show()

# Plot Feature Importance for XGBoost
plt.figure(figsize=(10, 5))
sns.barplot(x="Importance", y="Feature", data=xgb_importance_df, palette="Oranges_r")
plt.title("Feature Importance (XGBoost)")
plt.show()

# Print Feature Importance Scores
print("\n🔹 Random Forest Feature Importance:")
print(rf_importance_df)

print("\n🔹 XGBoost Feature Importance:")
print(xgb_importance_df)
