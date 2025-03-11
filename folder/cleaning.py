import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import shutil
import stat


# Define error handler to clear read-only attribute and retry deletion
def remove_readonly(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)


# Set the master folder containing all fault type subfolders.
master_dir = r"C:\Users\shaur\OneDrive\Desktop\Fault_Data"  # Update to your master folder path

# Define fixed number of points for resampling
NUM_POINTS = 1000

# Get all fault type folders (e.g., SK_Biasing, SK_Drifts, etc.)
fault_folders = [d for d in os.listdir(master_dir) if os.path.isdir(os.path.join(master_dir, d))]

if not fault_folders:
    print("❌ No fault type folders found in the master directory!")
else:
    print(f"✅ Found {len(fault_folders)} fault type folders. Processing...")

for fault in fault_folders:
    # Define data directory (raw_files) and output directory (cleaned_files) for each fault folder.
    data_dir = os.path.join(master_dir, fault, "raw_files")
    output_dir = os.path.join(master_dir, fault, "cleaned_files")

    # Remove the existing cleaned_files folder if it exists, then recreate it.
    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir, onerror=remove_readonly)
        except Exception as e:
            print(f"❌ Could not delete {output_dir}: {e}")
    os.makedirs(output_dir, exist_ok=True)

    # List all CSV/Excel data files in the raw_files folder.
    data_files = [f for f in os.listdir(data_dir) if f.endswith(".csv") or f.endswith(".xlsx")]

    if not data_files:
        print(f"❌ No data files found in {data_dir}!")
        continue
    else:
        print(f"✅ Found {len(data_files)} files in {data_dir}. Processing...")

    for file in data_files:
        try:
            file_path = os.path.join(data_dir, file)

            # Read the file based on its format.
            if file.endswith(".csv"):
                df = pd.read_csv(file_path)
            elif file.endswith(".xlsx"):
                df = pd.read_excel(file_path)
            else:
                print(f"❌ Skipping {file}: Unsupported file format")
                continue

            # Drop empty columns.
            df = df.dropna(axis=1, how='all')

            # Identify column names dynamically.
            column_names = df.columns.tolist()
            time_cols = [col for col in column_names if "Time" in col]
            vin_col = [col for col in column_names if "Vin" in col]
            vout_col = [col for col in column_names if "Vout" in col]

            # Ensure necessary columns exist.
            if not time_cols or not vin_col or not vout_col:
                print(f"❌ Skipping {file}: Missing required columns!")
                continue

            # Select correct columns dynamically in required order: Time | Vin | Vout.
            df = df[[time_cols[0], vin_col[0], vout_col[0]]]
            df.columns = ["Time", "Vin", "Vout"]

            # Ensure the time column is sorted.
            df = df.sort_values(by="Time").reset_index(drop=True)

            # Interpolate to a fixed number of points (NUM_POINTS).
            interp_time = np.linspace(df["Time"].min(), df["Time"].max(), NUM_POINTS)
            interp_vin = interp1d(df["Time"], df["Vin"], kind='linear', fill_value="extrapolate")(interp_time)
            interp_vout = interp1d(df["Time"], df["Vout"], kind='linear', fill_value="extrapolate")(interp_time)

            # NOTE: Normalization has been removed to preserve the raw signal characteristics.

            # Create the cleaned dataframe.
            cleaned_df = pd.DataFrame({
                "Time": interp_time,
                "Vin": interp_vin,
                "Vout": interp_vout
            })

            # Save cleaned CSV.
            cleaned_file_path = os.path.join(output_dir, f"cleaned_{os.path.splitext(file)[0]}.csv")
            cleaned_df.to_csv(cleaned_file_path, index=False)

            print(f"✅ Processed and saved: {cleaned_file_path}")

        except Exception as e:
            print(f"❌ Error processing {file}: {e}")

print("🎉 All fault files cleaned!")
