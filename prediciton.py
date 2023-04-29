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
    for device in ["Glass", "Phone", "Watch"]:
        device_file = os.path.join("RollingDataFinalTest", f"User{person}_{device}_rolling.csv")
        if os.path.isfile(device_file):
            device_data = pd.read_csv(device_file)
            # Add suffix to column names
            suffix = f"_{device.split(' ')[0].lower()}"
            device_data = device_data.add_suffix(suffix)
            train_data = train_data.append(device_data)

# Convert timestamp to seconds and normalize
# print(train_data["timestamp_glass"])
train_data["timestamp_glass"] = train_data["timestamp_glass"].apply(lambda x:str(x).rsplit('.', 1)[0])
# print("type99", train_data["timestamp_glass"][0])
train_data["timestamp_glass"] = train_data["timestamp_glass"].apply(lambda x: "00:00:00" if x == 'nan' else parse(x))
# train_data["timestamp_glass"] = train_data["timestamp_glass"].apply(lambda x: datetime.strptime(str(x).rsplit('.', 1)[0].strip(), '%H:%M:%S'))
train_data["timestamp_glass"] = pd.to_datetime(train_data["timestamp_glass"])
train_data["timestamp_glass"] = (train_data["timestamp_glass"] - train_data["timestamp_glass"].min()) / (train_data["timestamp_glass"].max() - train_data["timestamp_glass"].min())

# train_data["timestamp_phone"] = train_data["timestamp_phone"].apply(lambda x: datetime.strptime(str(x).rsplit('.', 1)[0].strip(), '%H:%M:%S'))
train_data["timestamp_phone"] = train_data["timestamp_phone"].apply(lambda x:str(x).rsplit('.', 1)[0])
# print("type99", train_data["timestamp_phone"][0])
train_data["timestamp_phone"] = train_data["timestamp_phone"].apply(lambda x: "00:00:00" if x == 'nan' else parse(x))
train_data["timestamp_phone"] = pd.to_datetime(train_data["timestamp_phone"])
train_data["timestamp_phone"] = (train_data["timestamp_phone"] - train_data["timestamp_phone"].min()) / (train_data["timestamp_phone"].max() - train_data["timestamp_phone"].min())

# train_data["timestamp_watch"] = train_data["timestamp_watch"].apply(lambda x: datetime.strptime(str(x).rsplit('.', 1)[0].strip(), '%H:%M:%S'))
train_data["timestamp_watch"] = train_data["timestamp_watch"].apply(lambda x:str(x).rsplit('.', 1)[0])
# print("type99", train_data["timestamp_watch"][0])
train_data["timestamp_watch"] = train_data["timestamp_watch"].apply(lambda x: "00:00:00" if x == 'nan' else parse(x))
train_data["timestamp_watch"] = pd.to_datetime(train_data["timestamp_watch"])
train_data["timestamp_watch"] = (train_data["timestamp_watch"] - train_data["timestamp_watch"].min()) / (train_data["timestamp_watch"].max() - train_data["timestamp_watch"].min())

# Split the data into input (X) and output (Y) variables
X_train = np.array(train_data[["x_window_phone", "y_window_phone", "x_window_watch", "y_window_watch", "x_window_glass", "y_window_glass"]])
Y_train = np.array(train_data[["x_window_glass", "y_window_glass", "z_window_glass"]])

# Reshape the input data into [samples, time steps, features]
X_train = X_train.reshape((X_train.shape[0], 1, num_features))





testPerson = 10
testdevice1 = "Phone"
test_data1 = pd.read_csv(os.path.join("RollingDataFinalTest", f"User{testPerson}_{testdevice1}_rolling.csv"))
suffix1 = f"_{testdevice1.split(' ')[0].lower()}"
test_data1 = test_data1.add_suffix(suffix1)
test_data = pd.DataFrame()
# print('test_data 1 777')
# print(test_data1)
test_data = test_data.append(test_data1)
# print(test_data)
testdevice2 = "Watch"
test_data2 = pd.read_csv(os.path.join("RollingDataFinalTest", f"User{testPerson}_{testdevice2}_rolling.csv"))
suffix2 = f"_{testdevice2.split(' ')[0].lower()}"
test_data2 = test_data2.add_suffix(suffix2)
test_data = test_data.append(test_data2)
# print('test_data 2 999', test_data2)
# print(test_data)
# Convert timestamp to seconds and normalize
# test_data["timestamp"] = pd.to_datetime(test_data["timestamp"])
# test_data["timestamp"] = test_data["timestamp"].apply(lambda x: datetime.strptime(x.split("days ")[1].rsplit('.', 1)[0], '%H:%M:%S'))
# test_data["timestamp"] = pd.to_datetime(test_data["timestamp"])

test_data["timestamp_phone"] = test_data["timestamp_phone"].apply(lambda x:str(x).rsplit('.', 1)[0])
# print("type99", test_data["timestamp_phone"][0])
test_data["timestamp_phone"] = test_data["timestamp_phone"].apply(lambda x: "00:00:00" if x == 'nan' else parse(x))
# test_data["timestamp_phone"] = test_data["timestamp_phone"].apply(lambda x: datetime.strptime(str(x).rsplit('.', 1)[0].strip(), '%H:%M:%S'))
test_data["timestamp_phone"] = pd.to_datetime(test_data["timestamp_phone"])
test_data["timestamp_phone"] = (test_data["timestamp_phone"] - test_data["timestamp_phone"].min()) / (test_data["timestamp_phone"].max() - test_data["timestamp_phone"].min())

test_data["timestamp_watch"] = test_data["timestamp_watch"].apply(lambda x:str(x).rsplit('.', 1)[0])
# print("type99", test_data["timestamp_watch"][0])
test_data["timestamp_watch"] = test_data["timestamp_watch"].apply(lambda x: "00:00:00" if x == 'nan' else parse(x))
# test_data["timestamp_watch"] = test_data["timestamp_watch"].apply(lambda x: datetime.strptime(str(x).rsplit('.', 1)[0].strip(), '%H:%M:%S'))
test_data["timestamp_watch"] = pd.to_datetime(test_data["timestamp_watch"])
test_data["timestamp_watch"] = (test_data["timestamp_watch"] - test_data["timestamp_watch"].min()) / (test_data["timestamp_watch"].max() - test_data["timestamp_watch"].min())


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
testdevice1 = "Phone"
test_data1 = pd.read_csv(os.path.join("RollingDataFinalTest", f"User{testPerson}_{testdevice1}_rolling.csv"))
suffix1 = f"_{testdevice1.split(' ')[0].lower()}"
test_data1 = test_data1.add_suffix(suffix1)
test_data = pd.DataFrame()
# print('test_data 1 777')
# print(test_data1)
test_data = test_data.append(test_data1)
# print(test_data)
testdevice2 = "Watch"
test_data2 = pd.read_csv(os.path.join("RollingDataFinalTest", f"User{testPerson}_{testdevice2}_rolling.csv"))
suffix2 = f"_{testdevice2.split(' ')[0].lower()}"
test_data2 = test_data2.add_suffix(suffix2)
test_data = test_data.append(test_data2)
# print('test_data 2 999', test_data2)
# print(test_data)
# Convert timestamp to seconds and normalize
# test_data["timestamp"] = pd.to_datetime(test_data["timestamp"])
# test_data["timestamp"] = test_data["timestamp"].apply(lambda x: datetime.strptime(x.split("days ")[1].rsplit('.', 1)[0], '%H:%M:%S'))
# test_data["timestamp"] = pd.to_datetime(test_data["timestamp"])

test_data["timestamp_phone"] = test_data["timestamp_phone"].apply(lambda x:str(x).rsplit('.', 1)[0])
# print("type99", test_data["timestamp_phone"][0])
test_data["timestamp_phone"] = test_data["timestamp_phone"].apply(lambda x: "00:00:00" if x == 'nan' else parse(x))
# test_data["timestamp_phone"] = test_data["timestamp_phone"].apply(lambda x: datetime.strptime(str(x).rsplit('.', 1)[0].strip(), '%H:%M:%S'))
test_data["timestamp_phone"] = pd.to_datetime(test_data["timestamp_phone"])
test_data["timestamp_phone"] = (test_data["timestamp_phone"] - test_data["timestamp_phone"].min()) / (test_data["timestamp_phone"].max() - test_data["timestamp_phone"].min())

test_data["timestamp_watch"] = test_data["timestamp_watch"].apply(lambda x:str(x).rsplit('.', 1)[0])
# print("type99", test_data["timestamp_watch"][0])
test_data["timestamp_watch"] = test_data["timestamp_watch"].apply(lambda x: "00:00:00" if x == 'nan' else parse(x))
# test_data["timestamp_watch"] = test_data["timestamp_watch"].apply(lambda x: datetime.strptime(str(x).rsplit('.', 1)[0].strip(), '%H:%M:%S'))
test_data["timestamp_watch"] = pd.to_datetime(test_data["timestamp_watch"])
test_data["timestamp_watch"] = (test_data["timestamp_watch"] - test_data["timestamp_watch"].min()) / (test_data["timestamp_watch"].max() - test_data["timestamp_watch"].min())

# Split the test data into input (X) and output (Y) variables
X_test = np.array(test_data[["x_window_phone", "y_window_phone", "x_window_watch", "y_window_watch"]])

# zeros_array = np.zeros((X_test.shape[0], 2))
# zeros_array = zeros_array.reshape((zeros_array.shape[0], 1, 2))



# # Reshape the input data into [samples, time steps, features]
# X_test = X_test.reshape((X_test.shape[0], 1, num_features - 2))
# X_test_zeros = np.concatenate((X_test, zeros_array), axis=1)
# X_test_reshaped = X_test_zeros.reshape((X_test.shape[0], 1, num_features))

zeros_array = np.zeros((X_test.shape[0], 2))
X_test_zeros = np.concatenate((X_test, zeros_array), axis=1)
X_test_reshaped = X_test_zeros.reshape((X_test.shape[0], 1, num_features))


# print("555 X_test_reshaped", X_test_reshaped)
# Make predictions on the test data
Y_pred = model.predict(X_test_reshaped)

# print("444 Y_pred", Y_pred)

# Combine the predicted values with the test data timestamps
Y_pred_df = pd.DataFrame(Y_pred, columns=["x_window.2", "y_window.2", "z_window"])
test_data_pred = pd.concat([test_data["timestamp_phone"], Y_pred_df], join='inner')
test_data_pred.to_csv("test_data_pred.csv", index=False)

