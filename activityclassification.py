import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load accelerometer data
data = pd.read_csv('linearAccelData.txt', header=None, names=['timestamp', 'x', 'y', 'z'])

# Load gyroscope data
gyro_data = pd.read_csv('gyroData.txt', header=None, names=['timestamp', 'x', 'y', 'z'])

# Combine accelerometer and gyroscope data
data['gyro_x'] = gyro_data['x']
data['gyro_y'] = gyro_data['y']
data['gyro_z'] = gyro_data['z']

# Label the data as walking or climbing
# You need to write the code to label the data based on your domain knowledge
data['label'] = ...

# Define features and labels
X = data.drop(['timestamp', 'label'], axis=1)
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
