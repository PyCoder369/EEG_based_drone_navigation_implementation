import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score


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
def preprocess_data_with_engineered_features(file_list, electrodes, label_column, scaling_method='standard'):
    data = load_and_engineer_features(file_list, electrodes, label_column)
    X = data.iloc[:, :-1].values  # Engineered features
    y = data.iloc[:, -1].values  # Labels (event column)

    # Select scaling method
    if scaling_method == 'standard':
        scaler = StandardScaler()
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
    elif scaling_method == 'robust':
        scaler = RobustScaler()

    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler


# 3. Train Random Forest Model
def train_random_forest_model(X_train, y_train):
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model


# 4. Evaluate Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"ROC-AUC: {roc_auc * 100:.2f}%")
    print("Classification Report:")
    print(report)
   #print("Confusion Matrix:")
   #print(matrix)


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
X_train, X_test, y_train, y_test, scaler = preprocess_data_with_engineered_features(file_list, top_electrodes,
                                                                                    label_column)

# Train the model
rf_model = train_random_forest_model(X_train, y_train)

# Evaluate the model
evaluate_model(rf_model, X_test, y_test)

# Test a specific row (example row from test set)
test_row_index = 12  # Replace with desired row index
test_file = "demoEEG/BRight.csv"
test_data = load_and_engineer_features([test_file], top_electrodes, label_column)
row_data = test_data.iloc[test_row_index, :-1].values
true_label = test_data.iloc[test_row_index, -1]
predict_row(rf_model, scaler, row_data, true_label)


import socket

ESP_IP = "192.168.129.119"  # Replace with the ESP32's IP address
ESP_PORT = 80
# Function to send command to ESP32
def send_to_esp(predicted_event):
    try:
        # Create a socket connection
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((ESP_IP, ESP_PORT))
            # Send the predicted event as a string
            command = str(predicted_event) + '\n'  # Add newline for ESP32 parsing
            s.sendall(command.encode())
            print(f"Sent event {predicted_event} to ESP32.")
    except Exception as e:
        print(f"Failed to send command to ESP32: {e}")

# Example usage: Replace this with your actual prediction code
predicted_event = 2  # Example prediction
send_to_esp(predicted_event)
