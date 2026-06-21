import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

df = pd.read_csv("dane.csv", header=None, names=["Date", "Open", "High", "Low", "Close", "Volume"])
data = df["Close"].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)


def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

sequence_length = 10
X, y = create_sequences(scaled_data, sequence_length)


split_index = int(0.7 * len(X))
X_train, y_train = X[:split_index], y[:split_index]
X_test, y_test = X[split_index:], y[split_index:]


model = Sequential([
    LSTM(64, input_shape=(sequence_length, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=20, batch_size=16)


train_pred = model.predict(X_train)
test_pred = model.predict(X_test)


train_pred_inv = scaler.inverse_transform(train_pred)
y_train_inv = scaler.inverse_transform(y_train)
test_pred_inv = scaler.inverse_transform(test_pred)
y_test_inv = scaler.inverse_transform(y_test)


plt.figure(figsize=(14, 6))


plt.subplot(1, 2, 1)
plt.plot(y_train_inv, label="Real")
plt.plot(train_pred_inv, label="Predicted")
plt.title("Train Prediction")
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(y_test_inv, label="Real")
plt.plot(test_pred_inv, label="Predicted")
plt.title("Test Prediction")
plt.legend()

plt.tight_layout()
plt.show()