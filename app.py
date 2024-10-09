from flask import Flask, request, jsonify
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import io
import numpy as np

# file_id = '1XMcD7d3Q7hjob5s0jSj2_wTbhX27lcHU'
# url = f'https://drive.google.com/uc?id={file_id}'
url = f'PulledFromDrive.xlsx'

df = pd.read_excel(url)

app = Flask(__name__)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, seq_length):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer (maps LSTM output to predicted prices)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))  # LSTM output
        out = self.fc(out[:, -1, :])     # Take the last output
        return out

def create_sequences(data, seq_length, prediction_length):
    X, y = [], []
    for i in range(len(data) - seq_length - prediction_length + 1):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length:i+seq_length+prediction_length, 0])
    return np.array(X), np.array(y)

seq_length = 3
prediction_length = 12
input_size = 1
hidden_size = 50
num_layers = 2
output_size = prediction_length
num_epochs = 1000
learning_rate = 0.001

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
        X, y = create_sequences(scaled_data, seq_length, prediction_length)

        # Convert to PyTorch tensors
        X_train = torch.from_numpy(X).float()
        y_train = torch.from_numpy(y).float()

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            predicted_sales = model(y_train[-1].view(1, 12, 1))
            predicted_sales = scaler.inverse_transform(predicted_sales.detach().numpy())

        # Return the predicted sales
        return jsonify({'predicted_sales': predicted_sales.tolist()}), 200
        # return jsonify({'message': 'Model trained successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # @app.route('/predict', methods=['POST'])
    # def predict():
    #     try:
    #         # Get data from request (e.g., user inputs the last 3 months data)
    #         data = request.get_json()
    #         months = data['months']  # Assumed input as list of months [last month, last but one, ...]

    #         # Preprocess and scale the data
    #         months = [[m] for m in months]  # Convert to 2D list
    #         months_scaled = scaler.transform(months)

    #         # Convert to PyTorch tensor
    #         months_tensor = torch.tensor(months_scaled, dtype=torch.float32).view(-1, 1)

    #         # Predict using the trained model
    #         model.eval()
    #         with torch.no_grad():
    #             predicted_sales = model(months_tensor)

    #         # Inverse scale the output to get original sales values
    #         predicted_sales_unscaled = scaler.inverse_transform(predicted_sales.numpy())

    #         # Return the predicted sales
    #         return jsonify({'predicted_sales': predicted_sales_unscaled.tolist()}), 200

    #     except Exception as e:
    #         return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)