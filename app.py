from flask import Flask, request, jsonify
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import io
import numpy as np

url = f'PulledFromDrive.xlsx'

df = pd.read_excel(url)

app = Flask(__name__)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, seq_length):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length

        # LSTM layer with dropout and bidirectional ******* 
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2, bidirectional=True)

        # Fully connected layer (adjust output size for bidirectional)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Multiply hidden size by 2 for bidirectional

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)  # Adjust h0 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)  # Adjust c0 for bidirectional

        out, _ = self.lstm(x, (h0, c0))  # LSTM output
        out = self.fc(out[:, -1, :])     # Take the last output
        return out

def create_sequences(data, seq_length, prediction_length):
    X, y = [], []
    for i in range(len(data) - seq_length - prediction_length + 1):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length:i + seq_length + prediction_length, 0])

    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Check if there are enough samples to split
    if len(X) < 2:  # Ensure there are at least 2 samples
        return X, None, y, None  # Return None for test sets if not enough data
    
    # Split data into 70% training and 30% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test

seq_length = 4
prediction_length = 12
input_size = 1
hidden_size = 90
num_layers = 2
output_size = prediction_length
num_epochs = 1900
learning_rate = 0.00098

model = LSTMModel(input_size, hidden_size, num_layers, output_size, seq_length)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['Sold(kg)']])

@app.route('/train', methods=['POST'])
def train_model():
    try:
        # Get CSV file from the request
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file provided'}), 400

        # Read the file as a CSV
        df = pd.read_csv(io.BytesIO(file.read()))
        
        # Extract features and target from CSV
        scaled_data = scaler.fit_transform(df[['Sold(kg)']])
        
        # Call create_sequences to get the training and testing sets
        X_train, X_test, y_train, y_test = create_sequences(scaled_data, seq_length, prediction_length)

        # If there are not enough samples for the test set, handle accordingly
        if X_test is None or y_test is None:
            return jsonify({'message': 'Not enough data for test set. Please provide more data.'}), 400

        # Convert to PyTorch tensors
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()
        X_test = torch.from_numpy(X_test).float()  # Convert test data to tensor

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Model evaluation and prediction
        model.eval()
        with torch.no_grad():
            # Get predictions for the last input in X_test
            last_input = X_test[-1].view(1, seq_length, input_size)  # Last input for prediction
            predicted_sales = model(last_input)  # Make prediction
            predicted_sales = scaler.inverse_transform(predicted_sales.detach().numpy())

            # Calculate accuracy based on the last test target
            last_actual = scaler.inverse_transform(y_test[-1].reshape(-1, 1))  # Last actual value
            accuracy = (1 - np.abs(predicted_sales - last_actual) / last_actual) * 100  # Calculate accuracy

        # Return the predicted sales and accuracy
        return jsonify({
            'predicted_sales': predicted_sales.tolist(),
            'accuracy': accuracy[0][0] if accuracy.size > 0 else None  # Return accuracy as a scalar
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
