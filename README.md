# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## THEORY
Recurrent Neural Networks (RNNs) are designed to work with sequential data by maintaining a "memory" of previous inputs using hidden states. This makes them ideal for tasks like time series prediction. The model takes a sequence of past stock prices and learns to predict the next price.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Import the required libraries and load the stock price dataset.

### STEP 2: 

Normalize the closing price values using MinMaxScaler.

### STEP 3: 

Create input sequences and output labels for the RNN using a sliding window approach.

### STEP 4: 

Define an RNN model using PyTorch's nn.Module with two layers and a hidden size of 64.

### STEP 5: 

Train the RNN using Mean Squared Error (MSE) loss and Adam optimizer over multiple epochs.

### STEP 6: 

Evaluate the model using the test data and visualize predictions against actual stock prices.



## PROGRAM
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Step 1: Load and Preprocess Data
df_train = pd.read_csv('trainset.csv')
df_test = pd.read_csv('testset.csv')

train_prices = df_train['Close'].values.reshape(-1, 1)
test_prices = df_test['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_prices)
scaled_test = scaler.transform(test_prices)

def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

seq_length = 60
x_train, y_train = create_sequences(scaled_train, seq_length)
x_test, y_test = create_sequences(scaled_test, seq_length)

x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

from torchinfo import summary
# input_size = (batch_size, seq_len, input_size)
summary(model, input_size=(64, 60, 1))

# Train the Model
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, epochs=20):
  train_losses = []
  model.train()
  for epoch in range(epochs):
    total_loss = 0
    for x_batch, y_batch in train_loader:
      x_batch,y_batch=x_batch.to(device),y_batch.to(device)
      optimizer.zero_grad()  # Clear previous gradients
      outputs = model(x_batch)  # Forward pass
      loss = criterion(outputs, y_batch)  # Compute loss
      loss.backward()  # Backpropagation
      optimizer.step()  # Update weights
      total_loss += loss.item()
    train_losses.append(total_loss / len(train_loader))
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}')
    # Plot training loss
  plt.plot(train_losses, label='Training Loss')
  plt.xlabel('Epoch')
  plt.ylabel('MSE Loss')
  plt.title('Training Loss Over Epochs BY; JEEVANESH')
  plt.legend()
  plt.show()

model.eval()
with torch.no_grad():
    predicted = model(x_test_tensor.to(device)).cpu().numpy()
    actual = y_test_tensor.cpu().numpy()

# Inverse transform the predictions and actual values
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(actual)

# Plot the predictions vs actual prices
plt.figure(figsize=(10, 6))
plt.plot(actual_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Stock Price Prediction using RNN BY: JEEVANESH')
plt.legend()
plt.show()
print("by JEEVANESH")
print(f'Predicted Price: {predicted_prices[-1]}')
print(f'Actual Price: {actual_prices[-1]}')

```
### OUTPUT

## Training Loss Over Epochs Plot
![image](https://github.com/user-attachments/assets/312c4d83-4394-4413-bf6f-20dd74ce0b2d)


## True Stock Price, Predicted Stock Price vs time
![image](https://github.com/user-attachments/assets/b837c4c7-b990-45eb-bc08-cedfecc13a8c)
![image](https://github.com/user-attachments/assets/75bd1938-181d-43d0-9d44-d6432cb9980a)


### Predictions
Predicted Price: [1092.2023]
Actual Price: [1115.65]

## RESULT
The RNN model was successfully developed and trained to predict stock prices. The model learned from the historical closing price data and was able to predict the future price with a close approximation to the actual value.
