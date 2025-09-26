import os
import numpy as np
import pandas as pd
from scipy.signal import cheby2, sosfiltfilt, find_peaks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib  # Import joblib for saving models
import matplotlib.pyplot as plt

# Define the directory containing separated stress and normal data
main_dir = r"C:\Users\Sruthi Vangari\Desktop\project"

# Lists to hold stress and normal features and labels
stress_features = []
normal_features = []

# Loop through the two folders: "Stress" and "Normal"
for label_folder in ['Stress', 'Normal']:
    subject_path = os.path.join(main_dir, label_folder)
    
    if os.path.exists(subject_path):
        print(f"Processing {label_folder} folder...")
        
        for csv_file in os.listdir(subject_path):
            csv_path = os.path.join(subject_path, csv_file)
            
            if csv_file.endswith('.csv'):
                print(f"  Found CSV file: {csv_file}")
                
                try:
                    signal = pd.read_csv(csv_path).values.flatten()
                    signal = pd.to_numeric(signal, errors='coerce')
                    signal = signal[~np.isnan(signal)]
                except Exception as e:
                    print(f"    Error reading {csv_path}: {e}")
                    continue
                
                if len(signal) == 0:
                    print(f"    Warning: {csv_path} is empty or could not be read properly.")
                    continue
                
                fs = 64
                num_whole_minutes = len(signal) // (fs * 60)
                if num_whole_minutes == 0:
                    print(f"    Skipping {csv_path} due to insufficient length.")
                    continue

                segments = np.array([signal[i * fs * 60:(i + 1) * fs * 60] for i in range(num_whole_minutes)])

                for segment in segments:
                    if np.isnan(segment).any():
                        print(f"    Skipping segment due to NaN values.")
                        continue
                    
                    fcutlow = 0.5
                    fcuthigh = 5
                    sos = cheby2(2, 20, [fcutlow, fcuthigh], btype='bandpass', fs=fs, output='sos')
                    filtered_PPG = sosfiltfilt(sos, segment)

                    pks, _ = find_peaks(filtered_PPG, distance=0.4 * fs, height=0)
                    locs = pks / fs
                    RRI = np.diff(locs)
                    RRI = RRI[(RRI > 0.5) & (RRI < 1.2)]

                    if len(RRI) == 0:
                        print(f"    Warning: No valid RRI found in segment.")
                        continue

                    feature_vector = [np.mean(RRI), np.std(RRI)]

                    if label_folder == 'Stress':
                        stress_features.append(feature_vector)
                    else:
                        normal_features.append(feature_vector)

# Convert to NumPy arrays
stress_features = np.array(stress_features)
normal_features = np.array(normal_features)

# Check if we have enough data for training
if len(stress_features) == 0 or len(normal_features) == 0:
    print("No sufficient data available for training. Please check the dataset.")
else:
    print(f"Total stress features: {stress_features.shape[0]}, Total normal features: {normal_features.shape[0]}")

    stress_labels = np.ones(stress_features.shape[0])
    normal_labels = np.zeros(normal_features.shape[0])

    X = np.vstack((stress_features, normal_features))
    y = np.hstack((stress_labels, normal_labels))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(solver='liblinear', random_state=42),
        'Linear SVC (for RFE)': LinearSVC(max_iter=10000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }

    model_names = []
    accuracy_scores = []
    f1_scores = []
    roc_auc_scores = []

    for model_name, model in models.items():
        print(f"\nTraining and selecting features with {model_name}...")

        if model_name == 'Linear SVC (for RFE)':
            selector = RFE(model, n_features_to_select=1, step=1)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
        else:
            selector = SelectKBest(score_func=f_classif, k='all')
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)

        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_test_selected)
        y_pred_proba = model.predict_proba(X_test_selected)[:, 1] if hasattr(model, 'predict_proba') else y_pred

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        print(f"Accuracy: {accuracy}")
        print(f"F1-score: {f1}")
        print(f"AUC-ROC: {roc_auc}")
        
        model_filename = f'{model_name.replace(" ", "_").lower()}_model.pkl'
        joblib.dump(model, model_filename)
        print(f"{model_name} saved as {model_filename}")

        model_names.append(model_name)
        accuracy_scores.append(accuracy)
        f1_scores.append(f1)
        roc_auc_scores.append(roc_auc)

    fig, ax = plt.subplots(figsize=(12, 8))

    bar_width = 0.25
    index = np.arange(len(model_names))

    bar1 = ax.bar(index, accuracy_scores, bar_width, label='Accuracy')
    bar2 = ax.bar(index + bar_width, f1_scores, bar_width, label='F1-score')
    bar3 = ax.bar(index + 2 * bar_width, roc_auc_scores, bar_width, label='AUC-ROC')

    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.show()