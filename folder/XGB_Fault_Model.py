# Import necessary libraries
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load dataset from CSV
df = pd.read_csv(r"C:\Users\shaur\OneDrive\Documents\preprocessed_time_features_extracted_dataset.csv")

# Separate features and labels
X = df.drop(columns=["label"])  # Features
y = df["label"]  # Target variable

# Encode labels to numerical values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Verify dataset shape
print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")


# TRAINING THE RF CLASSIFIER


# Train XGBoost model
xgb_model = xgb.XGBClassifier(objective="multi:softmax", num_class=len(set(y_train)),
                              max_depth=6, eta=0.3, n_estimators=100)
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate performance
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {accuracy_xgb:.4f}")

# Detailed classification report
print(classification_report(y_test, y_pred_xgb, target_names=label_encoder.classes_))
