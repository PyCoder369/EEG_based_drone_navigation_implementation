import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load and Engineer Features
def load_and_engineer_features(file_list, electrodes, label_column):
    combined_data = []
    for file in file_list:
        data = pd.read_csv(file)
        filtered_data = data.iloc[:, electrodes]
        engineered_features = pd.DataFrame({
            'mean': filtered_data.mean(axis=1),
            'var': filtered_data.var(axis=1),
            'skew': filtered_data.skew(axis=1)
        })
        engineered_features['label'] = data.iloc[:, label_column]
        combined_data.append(engineered_features)
    return pd.concat(combined_data, ignore_index=True)

# 2. Preprocess Data
def preprocess_data_with_engineered_features(file_list, electrodes, label_column):
    data = load_and_engineer_features(file_list, electrodes, label_column)
    X = data.iloc[:, :-1].values  # Engineered features
    y = data.iloc[:, -1].values  # Labels (event column)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler

# 3. Train SVM Model
def train_svm_model(X_train, y_train):
    svm_model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
    svm_model.fit(X_train, y_train)
    return svm_model

# 4. Evaluate Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(matrix)

# 5. Predict a Specific Row
def predict_row(model, scaler, row_data, true_label):
    row_scaled = scaler.transform([row_data])
    prediction = model.predict(row_scaled)
    print(f"Row Data: {row_data}")
    print(f"True Event: {true_label}")
    print(f"Predicted Event: {prediction[0]}")
    print("Prediction is", "CORRECT" if prediction[0] == true_label else "INCORRECT")
    return prediction[0]

# File paths
file_list = [
    "Test Files/demofileA.csv", "Test Files/demofileAnu.csv", "Test Files/demofileB.csv",
    "Test Files/demofileBG.csv", "Test Files/demofileDi.csv", "Test Files/demofileG.csv",
    "Test Files/demofileI.csv", "Test Files/demofileIn.csv", "Test Files/demofileM.csv",
    "Test Files/demofileN.csv", "Test Files/demofileP.csv", "Test Files/demofilesh.csv",
    "Test Files/demofileSu.csv", "Test Files/demofileT.csv", "Test Files/demofileV.csv"
]

# Top electrodes (select the electrodes you want to use)
top_electrodes = [13, 9, 12, 6]  # Replace with desired electrodes
label_column = 19  # Event label column

# Load, preprocess data
X_train, X_test, y_train, y_test, scaler = preprocess_data_with_engineered_features(file_list, top_electrodes, label_column)

# Train the model
svm_model = train_svm_model(X_train, y_train)

# Evaluate the model
evaluate_model(svm_model, X_test, y_test)

# Test a specific row (example row from test set)
test_row_index = 2001  # Replace with desired row index
test_file = "Test Files/demofileA.csv"
test_data = load_and_engineer_features([test_file], top_electrodes, label_column)
row_data = test_data.iloc[test_row_index, :-1].values
true_label = test_data.iloc[test_row_index, -1]
predict_row(svm_model, scaler, row_data, true_label)
