import os
import pandas as pd
import numpy as np
import zipfile

# Define paths and constants
zip_path = 'D:\\b21ci018\\Data_29_subjects.zip'
extract_path = 'D:\\b21ci018\\Data_29_subjects\\Subjects'
stress_threshold = 0.5  # Define your threshold for stress vs normal

# Unzip the dataset if not already extracted
if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Extracted files to {extract_path}")

def label_bvp_data(df, threshold=0.5):
    df['label'] = np.where(df['A'] > threshold, 'stress', 'normal')
    return df

# Process each subject's BVP file
for subject_folder in os.listdir(extract_path):
    subject_path = os.path.join(extract_path, subject_folder)
    bvp_path = os.path.join(subject_path, 'bvp.csv')
    labeled_path = os.path.join(subject_path, 'bvp_labeled.csv')

    if os.path.isfile(bvp_path):
        print(f"Processing file: {bvp_path}")

        # Load BVP data
        df = pd.read_csv(bvp_path, header=None)
        df.columns = ['A']  # Rename column to 'A'

        # Apply labeling
        df_labeled = label_bvp_data(df, stress_threshold)
        
        # Save the labeled data
        df_labeled.to_csv(labeled_path, index=False)
        print(f"Labeled file saved as: {labeled_path}")
    else:
        print(f"BVP file not found for subject in: {subject_path}")

print("All data processed.")
