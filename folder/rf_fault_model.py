import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv(r"C:\Users\shaur\OneDrive\Documents\preprocessed_time_features_extracted_dataset.csv")

# Separate features and labels
X = df.drop(columns=["label"])  # Features
y = df["label"]  # Target variable

# Encode labels (Random Forest & XGBoost require numeric labels)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Verify data shapes
print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")

# TRAINING THE RF CLASSIFIER

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Initialize and train Random Forest model
rf_model = RandomForestClassifier(n_estimators=500, max_features="sqrt", random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate performance
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.4f}")

# Detailed classification report
print(classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_))

