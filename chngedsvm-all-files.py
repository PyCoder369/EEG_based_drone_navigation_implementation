import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from scipy.signal import butter, filtfilt


# Function to apply bandpass filter and compute bandpower
def bandpower(x, fs, band, window_sec=None):
    low, high = band
    nyquist = 0.5 * fs
    low = low / nyquist
    high = high / nyquist
    b, a = butter(2, [low, high], btype='band')
    filtered_x = filtfilt(b, a, x)
    power = np.square(filtered_x).mean()
    return power


# Function to create sliding window features and assign labels
def sliding_window_with_labels(data, labels, window_size, step_size):
    """
    Create sliding window features and corresponding labels
    Args:
        data: EEG data (numpy array)
        labels: Label array (numpy array)
        window_size: Size of the window in samples
        step_size: Step size for the sliding window in samples
    Returns:
        X_windowed: Windowed data (numpy array)
        y_windowed: Corresponding labels (numpy array)
    """
    windows = []
    window_labels = []
    for start in range(0, len(data) - window_size, step_size):
        windows.append(data[start:start + window_size])
        # Assign label based on majority class in the window
        window_labels.append(np.bincount(labels[start:start + window_size]).argmax())
    return np.array(windows), np.array(window_labels)


# Path to folder containing all CSV files
folder_path = 'Test Files'  # Update with the correct folder path

# Hyperparameter grid for RandomizedSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100, 1000],
    'gamma': [0.001, 0.01, 0.1],
    'kernel': ['linear', 'rbf'],
}

# Set the window size and step size for sliding window
window_size = 250  # in samples
step_size = 30  # in samples

# Bandpower frequency bands (Delta, Theta, Alpha, Beta, Gamma)
bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta': (12, 30),
    'gamma': (30, 40),
}
8
# Cross-validation strategy using Stratified K-Fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Loop through all CSV files in the folder
electrode_accuracies_all_files = []

for csv_file in os.listdir(folder_path):
    if csv_file.endswith('.csv'):
        print(f"Processing {csv_file}...")

        # Read the CSV file
        df = pd.read_csv(os.path.join(folder_path, csv_file))

        # Impute missing values in the first 19 columns (electrodes)
        imputer = SimpleImputer(strategy='mean')
        df.iloc[:, :19] = imputer.fit_transform(df.iloc[:, :19])

        # Extract labels (the 20th column for events)
        y = df.iloc[:, 19].values  # Labels (events)

        electrode_accuracies = []

        # Loop through each electrode (first 19 columns)
        for electrode in range(19):
            X = df.iloc[:, electrode].values  # Features for this electrode

            # Create temporal features using sliding window and assign labels
            X_windowed, y_windowed = sliding_window_with_labels(X, y, window_size, step_size)

            # Compute bandpower features for different frequency bands
            bandpowers = []
            for band_name, band_range in bands.items():
                bandpower_features = [bandpower(window, fs=256, band=band_range) for window in X_windowed]
                bandpowers.append(np.array(bandpower_features))

            # Combine bandpower features into a single feature set
            X_combined = np.vstack(bandpowers).T

            # Standardize the data for this electrode
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_combined)

            # Stratified K-Fold cross-validation
            fold_accuracies = []

            for train_index, test_index in kf.split(X_scaled, y_windowed):
                X_train, X_test = X_scaled[train_index], X_scaled[test_index]
                y_train, y_test = y_windowed[train_index], y_windowed[test_index]

                # Initialize SVM model
                svm_model = SVC(random_state=42)

                # RandomizedSearchCV for hyperparameter tuning
                random_search = RandomizedSearchCV(
                    svm_model,
                    param_distributions=param_grid,
                    n_iter=10,  # Number of parameter settings to sample
                    cv=3,  # 3-fold CV inside the random search
                    scoring='accuracy',
                    n_jobs=-1,
                    random_state=42
                )
                random_search.fit(X_train, y_train)

                # Get the best hyperparameters and the final model
                best_params = random_search.best_params_
                best_svm = random_search.best_estimator_

                # Train and test the model
                best_svm.fit(X_train, y_train)
                y_pred = best_svm.predict(X_test)

                # Calculate accuracy
                accuracy = accuracy_score(y_test, y_pred) * 100
                fold_accuracies.append(accuracy)

            # Calculate average accuracy across all folds
            avg_accuracy = np.mean(fold_accuracies)
            electrode_accuracies.append(avg_accuracy)

            print(f"Electrode {electrode + 1}: Average Accuracy = {avg_accuracy:.2f}%")

        # Calculate and store results for this file
        file_average_accuracy = np.mean(electrode_accuracies)
        electrode_accuracies_all_files.append(file_average_accuracy)
        print(f"Average accuracy for {csv_file}: {file_average_accuracy:.2f}%")

# Display overall results
overall_average_accuracy = np.mean(electrode_accuracies_all_files)
print(f"\nOverall average accuracy across all files: {overall_average_accuracy:.2f}%")
