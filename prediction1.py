import os
import pandas as pd
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from dateutil.parser import parse

# Parameters for sliding window
window_size = 100 # in milliseconds
stride = 50 # in milliseconds
num_features = 6 # 2 for phone, 2 for watch, and 2 for glass

# Load training data
train_data = pd.DataFrame()
for person in range(1, 18):
    for device in ["Glass", "HTC - front", "Samsung - back"]:
        device_file = os.path.join("rollingData", f"rollingData_person{person}_{device}.csv")
        if os.path.isfile(device_file):
            device_data = pd.read_csv(device_file)
            train_data = train_data.append(device_data)

# Convert timestamp to seconds and normalize
train_data["timestamp"] = train_data["timestamp"].apply(lambda x: (parse(x) - datetime(1970,1,1)).total_seconds())
train_data["timestamp"] = (train_data["timestamp"] - train_data["timestamp"].min()) / (train_data["timestamp"].max() - train_data["timestamp"].min())

# Split the data into input (X) and output (Y) variables
X_train = np.array(train_data[["x_window", "y_window", "x_window.1", "y_window.1"]])
Y_train = np.array(train_data[["x_window.2", "y_window.2", "z_window"]])

# Reshape the input data into [samples, time steps, features]
X_train = X_train.reshape((X_train.shape[0], 1, num_features))

# Define the model architecture
model = Sequential()
model.add(LSTM(64, input_shape=(1, num_features), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(16))
model.add(Dense(3, activation="linear"))

# Compile the model
model.compile(loss="mean_squared_error", optimizer="adam")

# Train the model
model.fit(X_train, Y_train, epochs=50, batch_size=64)

# Load test data
testPerson = 10
test_data = pd.read_csv(os.path.join("rollingData", "rollingData_person"+ str(testPerson) +"_HTC - front.csv"))

# Convert timestamp to seconds and normalize
test_data["timestamp"] = test_data["timestamp"].apply(lambda x: (parse(x) - datetime(1970,1,1)).total_seconds())
test_data["timestamp"] = (test_data["timestamp"] - test_data["timestamp"].min()) / (test_data["timestamp"].max() - test_data["timestamp"].min())

# Split the test data into input (X) and output (Y) variables
X_test = np.array(test_data[["x_window", "y_window", "x_window.1", "y_window.1"]])

# Reshape the input data into [samples, time steps, features]
X_test = X_test.reshape((X_test.shape[0], 1, num_features))

# Make
