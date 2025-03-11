import os
import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------------------------
# Master folder containing all fault folders
BASE_FOLDER = r"C:\Users\shaur\OneDrive\Desktop\Fault_Data"

# Augmentation parameters
DEFAULT_NUM_AUGMENTS = 5  # Minimum number of augmented versions to generate per cleaned file
NOISE_LEVEL = 0.01  # Standard deviation for Gaussian noise (as fraction of signal amplitude)
AMPLITUDE_SCALE_MIN = 0.95  # Minimum scaling factor for amplitude
AMPLITUDE_SCALE_MAX = 1.05  # Maximum scaling factor for amplitude
TIME_SHIFT_MAX = 0.01  # Maximum time shift (same units as Time column)

# Desired total samples per class (including originals + augmented files)
TARGET_SAMPLES_PER_CLASS = 100

# Set to True to dynamically compute number of augmentations per file if samples are low
DYNAMIC_AUGMENTATION = True


# ------------------------------------------------------------------------------
# 2. Augmentation Functions (Raw Signal Level)
# ------------------------------------------------------------------------------
def add_gaussian_noise(signal, noise_level=NOISE_LEVEL):
    """Adds Gaussian noise to a signal."""
    noise_std = noise_level * np.mean(np.abs(signal))
    noise = np.random.normal(0, noise_std, size=signal.shape)
    return signal + noise


def apply_amplitude_scaling(signal, scale_min=AMPLITUDE_SCALE_MIN, scale_max=AMPLITUDE_SCALE_MAX):
    """Scales the amplitude of a signal by a random factor."""
    scale_factor = np.random.uniform(scale_min, scale_max)
    return signal * scale_factor


def apply_time_shift(time, max_shift=TIME_SHIFT_MAX):
    """Applies a constant time shift to the time vector."""
    shift = np.random.uniform(-max_shift, max_shift)
    return time + shift


def augment_raw_data(time, vin, vout):
    """
    Applies augmentation techniques to raw signals.
    Returns augmented (time, vin, vout).
    """
    # Apply Gaussian noise to Vin and Vout
    vin_aug = add_gaussian_noise(vin)
    vout_aug = add_gaussian_noise(vout)

    # Apply amplitude scaling
    vin_aug = apply_amplitude_scaling(vin_aug)
    vout_aug = apply_amplitude_scaling(vout_aug)

    # Apply time shift to Time
    time_aug = apply_time_shift(time)

    return time_aug, vin_aug, vout_aug


# ------------------------------------------------------------------------------
# 3. Main Script: Augment Files from Cleaned Data
# ------------------------------------------------------------------------------
def process_and_augment_files():
    """
    Iterates through each fault folder, processes cleaned Excel/CSV files from the 'cleaned_files'
    subfolder, and performs augmentation.
    Saves augmented files in 'augmented_files' subfolders within each fault folder.
    """
    fault_folders = [d for d in os.listdir(BASE_FOLDER) if os.path.isdir(os.path.join(BASE_FOLDER, d))]

    if not fault_folders:
        print("❌ No fault type folders found in the master directory!")
        return
    else:
        print(f"✅ Found {len(fault_folders)} fault type folders. Processing augmentation...")

    for fault_folder in fault_folders:
        cleaned_folder = os.path.join(BASE_FOLDER, fault_folder, "cleaned_files")
        augmented_folder = os.path.join(BASE_FOLDER, fault_folder, "augmented_files")

        # Ensure the augmented_files folder exists
        os.makedirs(augmented_folder, exist_ok=True)

        # List all cleaned files (both .csv and .xlsx) in the cleaned_files folder
        cleaned_files = [f for f in os.listdir(cleaned_folder) if f.endswith((".csv", ".xlsx"))] if os.path.exists(
            cleaned_folder) else []

        if not cleaned_files:
            print(f"❌ No cleaned files found in {cleaned_folder}! Skipping folder {fault_folder}.")
            continue
        else:
            print(f"✅ Processing folder: {fault_folder} - found {len(cleaned_files)} cleaned files.")

        # Determine the number of augmentations per file
        if DYNAMIC_AUGMENTATION:
            required_augments = int(np.ceil((TARGET_SAMPLES_PER_CLASS / len(cleaned_files)) - 1))
            num_augments = max(DEFAULT_NUM_AUGMENTS, required_augments)
            print(
                f"For folder {fault_folder}, using {num_augments} augmentations per file to reach at least {TARGET_SAMPLES_PER_CLASS} samples.")
        else:
            num_augments = DEFAULT_NUM_AUGMENTS

        # Process each cleaned file for augmentation
        for cleaned_file in cleaned_files:
            try:
                file_path = os.path.join(cleaned_folder, cleaned_file)
                if cleaned_file.endswith(".xlsx"):
                    df = pd.read_excel(file_path)
                elif cleaned_file.endswith(".csv"):
                    df = pd.read_csv(file_path)

                # Ensure the required columns exist
                if not all(col in df.columns for col in ["Time", "Vin", "Vout"]):
                    print(f"❌ Skipping {cleaned_file}: Missing required columns.")
                    continue

                time_arr = df["Time"].values
                vin_arr = df["Vin"].values
                vout_arr = df["Vout"].values

                base_filename = os.path.splitext(cleaned_file)[0]

                # Generate augmented copies for each cleaned file
                for i in range(num_augments):
                    time_aug, vin_aug, vout_aug = augment_raw_data(time_arr, vin_arr, vout_arr)
                    df_aug = pd.DataFrame({
                        "Time": time_aug,
                        "Vin": vin_aug,
                        "Vout": vout_aug
                    })
                    aug_filename = f"{base_filename}_aug{i + 1}.csv"  # Save as CSV
                    aug_path = os.path.join(augmented_folder, aug_filename)
                    df_aug.to_csv(aug_path, index=False)  # Save as CSV
                print(f"✅ Augmented {base_filename} with {num_augments} versions.")
            except Exception as e:
                print(f"❌ Error processing {cleaned_file}: {e}")


if __name__ == "__main__":
    process_and_augment_files()
    print("🎉 Data augmentation completed.")
