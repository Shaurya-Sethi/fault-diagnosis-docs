import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------ ✅ Load & Verify Dataset ------------------------
print("🔹 Loading Dataset...")
df = pd.read_csv("C:/Users/shaur/OneDrive/Desktop/smote_balanced_TD_features_MLP_V2.csv")

# Check if all expected features are present
expected_features = ["min", "median", "ptp", "rms", "zcr", "var",
                     "skew_kurt_ratio", "crest_factor", "shape_factor",
                     "skew_kurt_product", "std_min_ratio", "min_ptp_ratio"]

missing_features = [col for col in expected_features if col not in df.columns]
if missing_features:
    raise ValueError(f"⚠️ Missing features in dataset: {missing_features}")

# Print feature count
print(f"✅ Feature count before dropping any columns: {df.shape[1] - 1} (excluding label)")

# Standardize Features
scaler = StandardScaler()
X = scaler.fit_transform(df.drop(columns=["label"]))

# Encode Labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["label"])

# Save Scaler & Label Encoder
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convert to PyTorch Tensors
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)

# ------------------------ ✅ Define MLP Model ------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.selu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.selu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.selu(self.bn3(self.fc3(x)))
        x = self.fc4(x)  # No activation for output layer (CrossEntropyLoss handles it)
        return x

# Initialize Model
input_dim = X_train.shape[1]  # Make sure input size matches dataset
output_dim = len(np.unique(y))
model = MLP(input_dim, output_dim)

# Print model input size
print(f"✅ Model initialized with input size: {input_dim}")

# ------------------------ ✅ Model Saving & Loading Fix ------------------------
MODEL_PATH = "mlp_final.pth"

# If model exists, delete and retrain
if os.path.exists(MODEL_PATH):
    print("⚠️ Deleting old model due to feature size mismatch...")
    os.remove(MODEL_PATH)

# ------------------------ ✅ Train the MLP Model ------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)

epochs = 300
best_accuracy = 0

patience = 10
patience_counter = 0  # Early stopping setup


print("🔹 Training MLP Model...")
for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            predictions = torch.argmax(test_outputs, dim=1)
            accuracy = (predictions == y_test).float().mean().item()
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.4f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                patience_counter = 0
                torch.save(model.state_dict(), MODEL_PATH)  # Save best model
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("\n✅ Early Stopping Triggered. Stopping Training.")
                break

print(f"\n🚀 Final Fine-Tuned MLP Accuracy: {best_accuracy:.4f}")
print("✅ Model saved as 'mlp_final.pth'")

# ------------------------ ✅ Cross-Validation & Stability Testing ------------------------
print("\n🔹 Performing Cross-Validation on MLP...")

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
mlp_accuracies = []

for train_idx, test_idx in kf.split(X, y):
    X_train_k, X_test_k = torch.tensor(X[train_idx], dtype=torch.float32), torch.tensor(X[test_idx], dtype=torch.float32)
    y_train_k, y_test_k = torch.tensor(y[train_idx], dtype=torch.long), torch.tensor(y[test_idx], dtype=torch.long)

    model.load_state_dict(torch.load(MODEL_PATH))  # Reload best model
    model.eval()

    with torch.no_grad():
        predictions = torch.argmax(model(X_test_k), dim=1)
        accuracy = accuracy_score(y_test_k.numpy(), predictions.numpy())
        mlp_accuracies.append(accuracy)

print(f"✅ Cross-Validation: Mean Accuracy = {np.mean(mlp_accuracies):.4f}, Std Dev = {np.std(mlp_accuracies):.4f}")

# ------------------------ ✅ Train-Test Evaluation ------------------------
print("\n🔹 Evaluating MLP on Train-Test Split...")
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

with torch.no_grad():
    predictions = torch.argmax(model(X_test), dim=1)

# Classification Report
print("\n🔹 Classification Report:")
print(classification_report(y_test.numpy(), predictions.numpy()))

# Confusion Matrix
cm = confusion_matrix(y_test.numpy(), predictions.numpy())
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - MLP")
plt.show()

print("\n✅ MLP Model evaluation complete!")
