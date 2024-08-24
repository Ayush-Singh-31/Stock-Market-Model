import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Load and preview the training data
train_data = pd.read_csv('Train.csv')
print(train_data.head())

# Extract the training data (column 1: "Open" prices)
train_prices = train_data.iloc[:, 1:2].values
print(train_prices)
print(train_prices.shape)

# Scale the training data to the range (0, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
train_prices_scaled = scaler.fit_transform(train_prices)
print(train_prices_scaled)

# Create data structures for LSTM input (X_train) and output (y_train)
X_train = []
y_train = []
for i in range(60, len(train_prices_scaled)):
    X_train.append(train_prices_scaled[i - 60:i, 0])
    y_train.append(train_prices_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
print(X_train)
print(y_train)

# Reshape the input data for LSTM [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print(X_train.shape)

# Build the LSTM model
model = Sequential()

# Add the first LSTM layer with Dropout regularization
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))

# Add additional LSTM layers with Dropout
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))

# Add the output layer
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Load and prepare the test data
test_data = pd.read_csv('Test.csv')
real_prices = test_data.iloc[:, 1:2].values

# Combine train and test data to get inputs for prediction
total_prices = pd.concat((train_data['Open'], test_data['Open']), axis=0)
input_data = total_prices[len(total_prices) - len(test_data) - 60:].values

# Reshape and scale the test inputs
input_data = input_data.reshape(-1, 1)
input_data = scaler.transform(input_data)

# Create the test data structure
X_test = []
for i in range(60, len(input_data)):
    X_test.append(input_data[i - 60:i, 0])
X_test = np.array(X_test)

# Reshape the test input data for LSTM
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Make predictions on the test data
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Visualize the results
plt.plot(real_prices, color='red', label='Real Stock Price')
plt.plot(predicted_prices, color='blue', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
