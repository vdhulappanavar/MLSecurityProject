import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Assuming you have the rolling window data stored in a pandas DataFrame
# where each row represents a sample and the rolling window data is in 
# columns "x_window", "y_window", and "z_window"
X = np.stack((combined_data["x_window"], combined_data["y_window"], combined_data["z_window"]), axis=-1)

# Assuming you have a corresponding array of labels
y = np.array(combined_data["activity"])

# Split data into train and test sets
train_split = 0.8
num_train = int(train_split * X.shape[0])
X_train, X_test = X[:num_train], X[num_train:]
y_train, y_test = y[:num_train], y[num_train:]

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
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
