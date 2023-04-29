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
    glass_file = os.path.join("data", "Training Data", f"User00{person}", "Glass", "linearAccelData.txt")
    if os.path.isfile(glass_file):
        glass_data = pd.read_csv(glass_file, header=None, names=["timestamp", "x", "y", "z"])

    phone_file = os.path.join("data", "Training Data", f"User00{person}", "HTC - front", "linearAccelDataM.txt")
    if os.path.isfile(phone_file):
        phone_data = pd.read_csv(phone_file, header=None, names=["timestamp", "x", "y", "z"])
        
    watch_file = os.path.join("data", "Training Data", f"User00{person}", "Samsung - back", "linearAccelDataM.txt")
    if os.path.isfile(watch_file):
        watch_data = pd.read_csv(watch_file, header=None, names=["timestamp", "x", "y", "z"])

    # Normalize the timestamps to start from 0
    time_format = "%Y-%m-%d %H:%M:%S:%f"
    for device, device_data in [("Glass", glass_data), ("Phone", phone_data), ("Watch", watch_data)]:
        if not device_data.empty:
            device_data["timestamp"] = device_data["timestamp"].apply(lambda x: datetime.strptime(x, time_format))
            device_data["timestamp"] = pd.to_datetime(device_data["timestamp"])
            device_data.sort_values(by=["timestamp"], inplace=True)
            device_data["timestamp"] = device_data["timestamp"] - device_data["timestamp"].iloc[0]
            device_data["timestamp"] = device_data["timestamp"].apply(lambda x: str(x).split("days ")[1])

            # Create a sliding window for each axis (x, y, z)
            for axis in ["x", "y", "z"]:
                def rolling_window(x):
                    if len(x) < window_size:
                        return np.nan
                    else:
                        return np.mean(x[-window_size:])

                # Create a new column for the axis with the rolling window data
                device_data[f"{axis}_window"] = device_data[axis].rolling(window=window_size, min_periods=1).apply(rolling_window, raw=True).copy()

            # Remove any rows with NaN values
            device_data.dropna(inplace=True)

            # Create a new data frame with the rolling window data and their respective timestamps
            rolling_data = pd.DataFrame({
                "timestamp": device_data["timestamp"],
                "x_window": device_data["x_window"],
                "y_window": device_data["y_window"],
                "z_window": device_data["z_window"],
                "device": device
            })

            # Create a new column for the person number
            rolling_data["person"] = person
            rolling_data.to_csv(f"RollingDataFinalTest/User{person}_{device}_rolling.csv", index=False)

