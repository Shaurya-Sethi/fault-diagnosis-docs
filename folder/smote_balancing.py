# smote_balancing.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load preprocessed dataset (original dataset remains unchanged)
df = pd.read_csv(r"C:\Users\shaur\OneDrive\Desktop\TD_features_MLP_V2.csv")

# Separate features and labels
X = df.drop(columns=["label"])  # Features
y = df["label"]  # Target variable

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split into train-test before SMOTE (stratified to preserve class proportions)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Check original class distribution in training set before SMOTE
unique_train, counts_train = np.unique(y_train, return_counts=True)
print("\nOriginal Training Class Distribution (Before SMOTE):")
for label, count in dict(zip(label_encoder.inverse_transform(unique_train), counts_train)).items():
    print(f"{label}: {count}")

# Apply SMOTE (Increase "SK_Normal" to 200 samples but keep faults unchanged)
smote = SMOTE(sampling_strategy={label_encoder.transform(['SK_Normal'])[0]: 200}, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Check new class distribution after SMOTE
unique_resampled, counts_resampled = np.unique(y_resampled, return_counts=True)
class_distribution = dict(zip(label_encoder.inverse_transform(unique_resampled), counts_resampled))

print("\nNew Class Distribution After SMOTE:")
for label, count in class_distribution.items():
    print(f"{label}: {count}")

# Convert resampled data into DataFrame
df_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
df_resampled["label"] = label_encoder.inverse_transform(y_resampled)  # Convert back to original labels

# save dataset
desktop_path = r"C:\Users\shaur\OneDrive\Desktop\smote_balanced_TD_features_MLP_V2.csv"

# Save the resampled dataset
df_resampled.to_csv(desktop_path, index=False)
print(f"\nSMOTE-balanced dataset saved at: {desktop_path}")

# Plot new class distribution
df_class_dist = pd.DataFrame({"Class": list(class_distribution.keys()), "Count": list(class_distribution.values())})
plt.figure(figsize=(10, 5))
sns.barplot(x="Class", y="Count", data=df_class_dist)
plt.xticks(rotation=45)
plt.title("Class Distribution After SMOTE")
plt.xlabel("Class Labels")
plt.ylabel("Number of Samples")
plt.show()
