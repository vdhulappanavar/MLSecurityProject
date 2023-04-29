import os
import numpy as np
import pandas as pd

# Define the window size and overlap
window_size = 100
overlap = 50

# Define a function to generate rolling data
def generate_rolling_data(data, label, device):
    # Apply rolling window function to the data
    rolling_data = data.rolling(window=window_size, min_periods=1).apply(lambda x: x[-window_size:]).dropna()
    # Get the number of windows that fit in the data with the specified overlap
    num_windows = int(np.ceil((rolling_data.shape[0] - window_size) / overlap)) + 1
    # Split the data into windows with the specified overlap
    windows = np.array_split(rolling_data, num_windows)
    # Label the windows with the activity and device
    labels = np.full((num_windows, 1), label)
    devices = np.full((num_windows, 1), device)
    # Combine the windows, activity, and device into one DataFrame
    combined_data = pd.concat([pd.DataFrame(np.concatenate(windows)),
                               pd.DataFrame(labels, columns=["activity"]),
                               pd.DataFrame(devices, columns=["device"])], axis=1)
    return combined_data

# Define the path to the Training Data directory
training_data_dir = "Training Data"

# Loop through the user directories and generate rolling data for each device
for i in range(1, 18):
    user = f"User00{i}"
    print(f"Generating rolling data for {user}")
    glass_data = pd.read_csv(os.path.join("data", training_data_dir, user, "Glass", "linearAccelData.txt"), header=None,
                             names=["timestamp", "x", "y", "z"])
    phone_data = pd.read_csv(os.path.join("data", training_data_dir, user, "HTC - front", "linearAccelDataM.txt"), header=None,
                             names=["timestamp", "x", "y", "z"])
    watch_data = pd.read_csv(os.path.join("data", training_data_dir, user, "Samsung - back", "linearAccelDataM.txt"), header=None,
                             names=["timestamp", "x", "y", "z"])
    # glass_data = generate_rolling_data(glass_data, "walking", "glass")
    # phone_data = generate_rolling_data(phone_data, "walking", "phone")
    # watch_data = generate_rolling_data(watch_data, "climbing", "watch")
    # # Concatenate the rolling data for all devices
    # combined_data = pd.concat([glass_data, phone_data, watch_data])
    # # Save the rolling data to a CSV file
    # combined_data.to_csv(f"{user}_RollingData.csv", index=False)
