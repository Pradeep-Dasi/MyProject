import os
import numpy as np
import pandas as pd
from scipy.signal import cheby2, sosfiltfilt, find_peaks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Define the main directory containing the subject folders
main_dir = r"D:\b21ci018\Data_29_subjects\Subjects"

# List to hold all features and labels
features = []
labels = []

# Check if directory exists
if not os.path.exists(main_dir):
    print(f"Error: The directory '{main_dir}' does not exist.")
else:
    # Loop through each subject folder
    for subject_folder in os.listdir(main_dir):
        subject_path = os.path.join(main_dir, subject_folder)
        if os.path.isdir(subject_path):  # Check if it is a directory
            print(f"Processing subject folder: {subject_folder}")

            # Look for the 'bvp_labeled.csv' file in the subject folder
            csv_path = os.path.join(subject_path, 'bvp_labeled.csv')
            if os.path.exists(csv_path):
                print(f"  Found CSV file: {csv_path}")

                # Load the CSV file
                try:
                    signal = pd.read_csv(csv_path).values.flatten()
                    
                    # Ensure all values in the signal are numeric
                    signal = pd.to_numeric(signal, errors='coerce')
                    signal = signal[~np.isnan(signal)]
                    
                except Exception as e:
                    print(f"    Error reading {csv_path}: {e}")
                    continue

                if len(signal) == 0:
                    print(f"    Warning: {csv_path} is empty or could not be read properly.")
                    continue  # Skip this file if empty

                # Define the sampling frequency
                fs = 64

                # Define segment lengths (in seconds)
                first_segment_length = 27 * 60
                last_segment_length = 5 * 60

                # Extract segments
                first_segment_end_index = int(first_segment_length * fs)
                last_segment_start_index = int((len(signal) / fs - last_segment_length) * fs)

                if first_segment_end_index >= len(signal) or last_segment_start_index <= 0:
                    print(f"    Skipping {csv_path} due to insufficient length.")
                    continue  # Skip if signal is too short

                first_segment = signal[:first_segment_end_index]
                last_segment = signal[last_segment_start_index:]
                middle_segment = signal[first_segment_end_index:last_segment_start_index]

                # Segment lengths
                num_whole_minutes = int(len(middle_segment) / fs / 60)
                if num_whole_minutes <= 0:
                    print(f"    Skipping {csv_path} due to insufficient middle segment.")
                    continue  # Skip if no full minute in the middle segment

                middle_matrix = np.array([middle_segment[i * fs * 60:(i + 1) * fs * 60] for i in range(num_whole_minutes)])

                first_matrix = np.array([first_segment[i * fs * 60:(i + 1) * fs * 60] for i in range(27)])
                last_matrix = np.array([last_segment[i * fs * 60:(i + 1) * fs * 60] for i in range(5)])

                # Concatenate the matrices
                all_segments = np.vstack((first_matrix, middle_matrix, last_matrix))

                # Add the labels (0 for rest, 1 for stress)
                pattern = np.ones((all_segments.shape[0], 1))
                pattern[[0, 1, 2, 14, 15, 21, 22, 26, 27]] = 0  # Rest periods
                pattern[[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 23, 24, 25]] = 1  # Stress periods
                pattern[-5:, :] = np.array([0, 0, 1, 0, 0]).reshape(-1, 1)  # Last 5 minutes pattern

                for j in range(all_segments.shape[0]):
                    current_segment = all_segments[j, :]

                    # Ensure the current segment contains only valid numeric data
                    if np.isnan(current_segment).any():
                        print(f"    Skipping segment {j} due to NaN values.")
                        continue

                    fcutlow = 0.5
                    fcuthigh = 5
                    sos = cheby2(2, 20, [fcutlow, fcuthigh], btype='bandpass', fs=fs, output='sos')
                    filtered_PPG = sosfiltfilt(sos, current_segment)

                    # Finding peaks
                    pks, _ = find_peaks(filtered_PPG, distance=0.4 * fs, height=0)
                    locs = pks / fs
                    RRI = np.diff(locs)
                    RRI = RRI[(RRI > 0.5) & (RRI < 1.2)]

                    # Feature extraction (mean and std of RRI)
                    if len(RRI) == 0:
                        print(f"    Warning: No valid RRI found in segment {j} of {csv_path}.")
                        continue

                    feature_vector = [np.mean(RRI), np.std(RRI)]
                    features.append(feature_vector)
                    labels.append(pattern[j])

    # Convert features and labels to NumPy arrays
    features = np.array(features)
    labels = np.array(labels).flatten()

    # Check if we have data
    if len(features) == 0 or len(labels) == 0:
        print("No data available for training. Please check the data processing steps.")
    else:
        print(f"Total features: {features.shape[0]}, Total labels: {labels.shape[0]}")

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Define models
        models = {
            'Logistic Regression': LogisticRegression(solver='liblinear', random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }

        # Train and evaluate each model
        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Evaluate the model
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            roc_auc = roc_auc_score(y_test, y_pred_proba)

            print(f"Accuracy: {accuracy}")
            print(f"F1-score: {f1}")
            print(f"AUC-ROC: {roc_auc}")