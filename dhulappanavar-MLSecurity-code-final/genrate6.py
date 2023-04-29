import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

window_size = 176

# Set the path to the directory containing the rolling data files
rolling_data_dir = "RollingDataFinalTest"

# Loop through the rolling data files for each user and concatenate the data
X_list = []
for i in range(1, 18):
    user = f"User{i}"
    glass_data = pd.read_csv(os.path.join(rolling_data_dir, f"{user}_Glass_rolling.csv"))
    phone_data = pd.read_csv(os.path.join(rolling_data_dir, f"{user}_Phone_rolling.csv"))
    watch_data = pd.read_csv(os.path.join(rolling_data_dir, f"{user}_Watch_rolling.csv"))
    # Combine the data into one DataFrame
    combined_data = pd.concat([glass_data, phone_data, watch_data], axis=1)
    # Get the rolling window data
    X_user = combined_data.values.reshape(-1, window_size, 3).astype('float32')
    X_list.append(X_user)

# Concatenate the data for all users
X = np.concatenate(X_list, axis=0)

# Split data into train and test sets
train_split = 0.8
num_train = int(train_split * X.shape[0])
X_train, X_test = X[:num_train], X[num_train:]

# Define the model architecture
model = tf.keras.Sequential([
    layers.Input(shape=(window_size, 3)),
    layers.Conv1D(filters=32, kernel_size=3, activation="relu"),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(6, activation="softmax")
])

# Compile the model
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Train the model
model.fit(X_train, epochs=10, batch_size=32, validation_data=(X_test,))
