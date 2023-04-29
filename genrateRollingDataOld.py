import os
import pandas as pd
from datetime import datetime
import numpy as np

# Parameters for sliding window
window_size = 100 # in milliseconds
stride = 50 # in milliseconds

# Iterate over all 17 persons
for person in range(1, 18):
    # Initialize data frames for each device
    glass_data = pd.DataFrame()
    phone_data = pd.DataFrame()
    watch_data = pd.DataFrame()

    # Load data from each device's linear acceleration file
    glass_file = os.path.join("data", "Training Data", f"User{person:03d}", "Glass", "linearAccelData.txt")
    if os.path.isfile(glass_file):
        glass_data = pd.read_csv(glass_file, header=None, names=["timestamp", "x", "y", "z"])

    phone_file = os.path.join("data", "Training Data", f"User{person:03d}", "HTC - front", "linearAccelDataM.txt")
    if os.path.isfile(phone_file):
        phone_data = pd.read_csv(phone_file, header=None, names=["timestamp", "x", "y", "z"])
    
    watch_file = os.path.join("data", "Training Data", f"User{person:03d}", "Samsung - back", "linearAccelDataM.txt")
    if os.path.isfile(watch_file):
        watch_data = pd.read_csv(watch_file, header=None, names=["timestamp", "x", "y", "z"])

    # print('watch_data', watch_data)
    # Create a dictionary to map the device name to the corresponding data frame
    device_data = {
        "glass": glass_data,
        "phone": phone_data,
        "watch": watch_data
    }

    time_format = "%Y-%m-%d %H:%M:%S:%f"

    # Iterate over each device and create rolling data separately
    for device_name, data in device_data.items():
        if not data.empty:
            # Convert the timestamp string to a datetime object
            data["timestamp"] = data["timestamp"].apply(lambda x: datetime.strptime(x, time_format))

            data["timestamp"] = pd.to_datetime(data["timestamp"])
            # Sort the data frame by timestamp
            data.sort_values(by=["timestamp"], inplace=True)

            # Normalize the timestamps to start from 0
            data["timestamp"] = data["timestamp"] - data["timestamp"].iloc[0]

            # Create a sliding window for each axis (x, y, z)
            for axis in ["x", "y", "z"]:
                def rolling_window(x):
                    if len(x) < window_size:
                        return np.nan
                    else:
                        return np.mean(x[-window_size:])
                # Create a new column for the axis with the rolling window data
                data[f"{axis}_window"] = data[axis].rolling(
                    window=window_size, min_periods=1).apply(
                        rolling_window, raw=True)

                # Remove any rows with NaN values
                data.dropna(inplace=True)

                # Create a new data frame with the rolling window data and their respective timestamps
                rolling_data = pd.DataFrame({
                    "timestamp": data["timestamp"],
                    f"{axis}_window": data[f"{axis}_window"]
                })

                # Create a new column for the person number and device name
                rolling_data["person"] = person
                rolling_data["device"] = device_name

                # Save the rolling data to a new file
                rolling_data.to_csv(f"rollingData/rollingData_person{person}_{device_name}_{axis}.csv", index=False)
        else:
            print('data empty', device_name)
