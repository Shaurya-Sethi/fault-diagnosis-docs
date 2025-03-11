import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib  # For saving scaler and label encoder
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------ Load and Preprocess Dataset ------------------------
print("🔹 Loading Dataset...")
df = pd.read_csv(r"C:\Users\shaur\OneDrive\Desktop\smote_balanced_TD_features_MLP_V2.csv")

# # Apply Final Feature Selection (Remove low-importance features like XGBoost & RF)
# df = df.drop(columns=["rms", "skew_kurt_ratio"])  # Aligning with previous feature selection

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

# ------------------------ Define MLP Model ------------------------
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
        self.dropout = nn.Dropout(0.3)  # Reduced Dropout for Better Learning

    def forward(self, x):
        x = F.selu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.selu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.selu(self.bn3(self.fc3(x)))
        x = self.fc4(x)  # No activation (Handled in CrossEntropyLoss)
        return x

# Initialize Model
input_dim = X_train.shape[1]
output_dim = len(np.unique(y))
model = MLP(input_dim, output_dim)

# ------------------------ Training Setup ------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=30, gamma=0.7)  # Reduce LR every 30 epochs

# ------------------------ Training Loop ------------------------
epochs = 300
best_accuracy = 0
patience, patience_counter = 10, 0  # Early stopping setup

print("🔹 Training MLP Model...")
for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            predictions = torch.argmax(test_outputs, dim=1)
            accuracy = (predictions == y_test).float().mean().item()
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.4f}")

            # Early Stopping
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                patience_counter = 0
                torch.save(model.state_dict(), "mlp_best.pth")  # Save best model
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("\n✅ Early Stopping Triggered. Stopping Training.")
                break

print(f"\n🚀 Final Fine-Tuned MLP Accuracy: {best_accuracy:.4f}")

# Save final model
torch.save(model.state_dict(), "mlp_final.pth")
print("✅ Model saved as 'mlp_final.pth'")

# ------------------------ Feature Importance Analysis ------------------------
print("🔹 Calculating Feature Importance...")

from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, ClassifierMixin

# ------------------------ PyTorch to Sklearn Wrapper ------------------------
class TorchMLPWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        """No-op fit function to satisfy sklearn API."""
        return self

    def predict(self, X):
        """Perform forward pass and return predictions."""
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            logits = self.model(X_tensor)
            return torch.argmax(logits, dim=1).numpy()  # Convert to NumPy

# ------------------------ Compute Permutation Importance ------------------------
def compute_permutation_importance(model, X_test, y_test):
    """Calculate permutation feature importance for an MLP."""
    sklearn_model = TorchMLPWrapper(model)  # Wrap MLP in sklearn API
    result = permutation_importance(
        estimator=sklearn_model,
        X=X_test,
        y=y_test,
        scoring="accuracy",
        n_repeats=10,
        random_state=42
    )
    return result.importances_mean

# Compute Feature Importance
print("🔹 Calculating Feature Importance...")
feature_importance = compute_permutation_importance(model, X_test.numpy(), y_test.numpy())

# Print feature importance
for i, importance in enumerate(feature_importance):
    print(f"Feature {i}: {importance:.4f}")


# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=df.drop(columns=["label"]).columns)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Permutation Feature Importance for MLP")
plt.show()

# ------------------------ Validation & Testing ------------------------
print("🔹 Validating Model...")
model.load_state_dict(torch.load("mlp_final.pth"))
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    y_pred = torch.argmax(y_pred, dim=1)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test.numpy(), y_pred.numpy()))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test.numpy(), y_pred.numpy())
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

print("✅ Model validation complete!")
