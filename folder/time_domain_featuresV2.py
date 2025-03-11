#!/usr/bin/env python3
"""
extract_time_features.py

This script re-computes time-domain features from the raw time-domain data files
found in both the 'cleaned_files' and 'augmented_files' subfolders of each fault folder.
It ensures that critical errors (such as negative peak-to-peak values or negative RMS)
are avoided by computing features directly on the original signal values.
The final merged dataset is saved as a CSV file for later use (e.g., for MLP training).

Assumed directory structure:
Fault_Data/
 ├── SK_Normal/
 │     ├── cleaned_files/
 │     │       file1.csv, file2.csv, ...
 │     └── augmented_files/
 │             file1_aug1.csv, file1_aug2.csv, ...
 ├── SK_Power/
 │     ├── cleaned_files/
 │     │       ...
 │     └── augmented_files/
 │             ...
 └── (other fault folders)
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

# Configuration
BASE_FOLDER = r"C:\Users\shaur\OneDrive\Desktop\Fault_Data"
OUTPUT_CSV = "merged_time_features_extracted.csv"

# Optionally, set a target sample count for resampling if desired (not used here)
TARGET_ROWS = 1000  # (if you wish to resample, you can uncomment the corresponding section)

def read_time_series(file_path):
    """
    Reads a CSV or Excel file and extracts the 'Time', 'Vin', and 'Vout' columns.
    Returns a DataFrame with these columns after converting them to numeric and dropping NaNs.
    """
    try:
        if file_path.lower().endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format.")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    # Dynamically find columns containing 'Time', 'Vin', and 'Vout'
    cols = df.columns.tolist()
    time_col = next((col for col in cols if "Time" in col), None)
    vin_col  = next((col for col in cols if "Vin" in col), None)
    vout_col = next((col for col in cols if "Vout" in col), None)

    if not time_col or not vin_col or not vout_col:
        print(f"Missing required columns in {file_path}")
        return None

    # Convert to numeric and drop rows with missing values
    df = df[[time_col, vin_col, vout_col]].copy()
    df.columns = ["Time", "Vin", "Vout"]
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df["Vin"]  = pd.to_numeric(df["Vin"], errors="coerce")
    df["Vout"] = pd.to_numeric(df["Vout"], errors="coerce")
    df.dropna(subset=["Time", "Vin", "Vout"], inplace=True)

    # (Optional: Resample to a fixed number of rows)
    # Uncomment below if you need to resample:
    # if len(df) >= 2:
    #     t_start, t_end = df["Time"].min(), df["Time"].max()
    #     new_time = np.linspace(t_start, t_end, TARGET_ROWS)
    #     df["Vin"] = np.interp(new_time, df["Time"], df["Vin"])
    #     df["Vout"] = np.interp(new_time, df["Time"], df["Vout"])
    #     df["Time"] = new_time

    return df

def compute_features(signal):
    """
    Computes a set of time-domain features from a 1D NumPy array 'signal'.
    Returns a dictionary of features.
    """
    # Ensure signal is a NumPy array of floats
    signal = np.array(signal, dtype=np.float64)
    # Basic statistics
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    max_val = np.max(signal)
    min_val = np.min(signal)
    median_val = np.median(signal)
    # Peak-to-peak: use absolute difference to be safe
    ptp_val = np.abs(max_val - min_val)
    # Skewness and kurtosis (handle cases where variance is zero)
    skew_val = skew(signal) if std_val > 0 else 0.0
    kurt_val = kurtosis(signal) if std_val > 0 else 0.0
    # RMS: sqrt(mean(square))
    rms_val = np.sqrt(np.mean(signal ** 2))
    # Zero Crossing Rate: number of times the sign changes divided by length
    # We use the standard approach: count sign changes and divide by (len(signal)-1)
    sign_changes = np.sum(np.abs(np.diff(np.sign(signal)))) / 2  # each crossing counted twice
    zcr_val = sign_changes / (len(signal) - 1) if len(signal) > 1 else 0

    features = {
        "mean": mean_val,
        "std": std_val,
        "max": max_val,
        "min": min_val,
        "median": median_val,
        "ptp": ptp_val,
        "skewness": skew_val,
        "kurtosis": kurt_val,
        "rms": rms_val,
        "zcr": zcr_val
    }
    return features

def process_file_features(file_path):
    """
    Reads a time-domain file (CSV/Excel), computes features from the Vout signal,
    and returns a dictionary of features.
    """
    df = read_time_series(file_path)
    if df is None or df.empty:
        return None
    # We'll compute features from Vout (you can add Vin if needed)
    vout = df["Vout"].values
    features = compute_features(vout)
    return features

def process_folder(folder_path):
    """
    Processes both 'cleaned_files' and 'augmented_files' subfolders in the given fault folder.
    Returns a list of dictionaries, one per file, including the file's label and source.
    """
    all_features = []
    for subfolder in ["cleaned_files", "augmented_files"]:
        subfolder_path = os.path.join(folder_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        # Process both CSV and Excel files
        files = [os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path)
                 if f.endswith((".csv", ".xlsx", ".xls"))]
        for file in files:
            feats = process_file_features(file)
            if feats is None:
                continue
            # Add metadata: label from folder name, and source (cleaned or augmented)
            feats["label"] = os.path.basename(folder_path)
            feats["source"] = subfolder
            feats["file"] = os.path.basename(file)
            all_features.append(feats)
    return all_features

def build_dataset(base_folder=BASE_FOLDER):
    """
    Iterates through each fault folder in the base folder, processes features from each file,
    and returns a merged DataFrame.
    """
    fault_folders = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
    dataset = []
    for fault in fault_folders:
        folder_path = os.path.join(base_folder, fault)
        features_list = process_folder(folder_path)
        if features_list:
            dataset.extend(features_list)
    if dataset:
        return pd.DataFrame(dataset)
    else:
        return pd.DataFrame()

if __name__ == "__main__":
    # Build the dataset from all fault folders.
    merged_dataset = build_dataset(BASE_FOLDER)
    if merged_dataset.empty:
        print("No features extracted. Please check your raw data.")
    else:
        # Optional: Drop columns that have too many missing values (e.g., energy_ratio, vin_vout_corr)
        # In this script, we don't compute those.
        # Save the merged dataset to CSV and Excel for mobile viewing.
        output_csv = "time_features_extracted_dataset.csv"
        output_excel = "time_features_extracted_dataset.xlsx"
        merged_dataset.to_csv(output_csv, index=False)
        merged_dataset.to_excel(output_excel, index=False)
        print(f"Extracted dataset saved to {output_csv} and {output_excel}.")
