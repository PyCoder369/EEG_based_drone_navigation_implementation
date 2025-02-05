import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. Data pre-processing
# Function to load and filter data
def load_and_filter_data(file_list, electrodes, label_column):
    combined_data = []
    for file in file_list:
        data = pd.read_csv(file)
        # Select only the specified electrodes and the event (label) column
        filtered_data = data.iloc[:, electrodes + [label_column]]
        combined_data.append(filtered_data)
    return pd.concat(combined_data, ignore_index=True)


# Preprocess Data
def preprocess_data(file_list, top_electrodes, label_column):
    # Load and filter the dataset
    data = load_and_filter_data(file_list, top_electrodes, label_column)

    # Separate features and labels
    X = data.iloc[:, :-1].values  # Features (selected electrodes)
    y = data.iloc[:, -1].values  # Labels (event column)

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler


# 2. Model Training
# Step 1: Train the SVM model
def train_svm_model(X_train, y_train):
    # Initialize the SVM model
    svm_model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)

    # Train the model
    svm_model.fit(X_train, y_train)

    return svm_model


# Step 2: Predict with the trained model
def predict_with_svm_model(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred


# Step 3: Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred) * 100

    # Generate classification report
    report = classification_report(y_test, y_pred)

    # Generate confusion matrix
    matrix = confusion_matrix(y_test, y_pred)

    print("SVM Model Evaluation:")
    print(f"Accuracy: {accuracy:.2f}%")
    print("Classification Report:")
    print(report)



# File paths
file_list = [
    "Test Files/demofileA.csv", "Test Files/demofileAnu.csv", "Test Files/demofileB.csv", "Test Files/demofileBG.csv",
    "Test Files/demofileDi.csv", "Test Files/demofileG.csv", "Test Files/demofileI.csv", "Test Files/demofileIn.csv",
    "Test Files/demofileM.csv", "Test Files/demofileN.csv", "Test Files/demofileP.csv", "Test Files/demofilesh.csv",
    "Test Files/demofileSu.csv", "Test Files/demofileT.csv", "Test Files/demofileV.csv"
]

# Electrode indices (adjusted to 0-based indexing for Python)
top_electrodes = [13, 9, 12, 6]  # Replace with your specific electrode indices
label_column = 19  # The column index for the event labels (20th column in 0-based indexing)

# Preprocess data
X_train, X_test, y_train, y_test, scaler = preprocess_data(file_list, top_electrodes, label_column)

# Train the SVM model
svm_model = train_svm_model(X_train, y_train)

# Evaluate the SVM model
evaluate_model(svm_model, X_test, y_test)
