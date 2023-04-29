import os
import csv

# files_path = [os.path.abspath('./data/Training Data') for x in os.listdir()]
# print(files_path)

# userLinerAccelData = {}
# for userfile in os.listdir('./data/Training _Data/'):
#     userLinerAccelData[userfile] = {}
#     for deviceFile in os.listdir('./data/Training _Data/'+userfile):
#         userLinerAccelData[userfile][deviceFile] = {}
#         with open('./data/Training _Data/'+userfile+'/linearAccelData.txt', newline='') as csvfile:
#             rows = csv.reader(csvfile, delimiter=' ', quotechar='|')



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
    glass_file = os.path.join("data", "Training Data", "User001", "Glass", "linearAccelData.txt")
    if os.path.isfile(glass_file):
        glass_data = pd.read_csv(glass_file, header=None, names=["timestamp", "x", "y", "z"])

    phone_file = os.path.join("data", "Training Data", "User001", "HTC - front", "linearAccelDataM.txt")
    if os.path.isfile(phone_file):
        phone_data = pd.read_csv(phone_file, header=None, names=["timestamp", "x", "y", "z"])
    watch_file = os.path.join("Directory", "Training Data", "User001", "Samsung - back", "linearAccelData.txt")
    if os.path.isfile(watch_file):
        watch_data = pd.read_csv(watch_file, header=None, names=["timestamp", "x", "y", "z"])

    # Combine all three data frames into one
    combined_data = pd.concat([glass_data, phone_data, watch_data], axis=0)

    time_format = "%Y-%m-%d %H:%M:%S:%f"

    # convert the timestamp string to a datetime object
    combined_data["timestamp"] = combined_data["timestamp"].apply(lambda x: datetime.strptime(x, time_format))

    combined_data["timestamp"] = pd.to_datetime(combined_data["timestamp"])
    # # Sort the combined data frame by timestamp
    combined_data.sort_values(by=["timestamp"], inplace=True)
    # print('combined_data["timestamp"]', type(combined_data["timestamp"]))

    # Normalize the timestamps to start from 0
    combined_data["timestamp"] = combined_data["timestamp"] - combined_data["timestamp"].iloc[0]

    # Create a sliding window for each axis (x, y, z)
    for axis in ["x", "y", "z"]:
        def rolling_window(x):
            if len(x) < window_size:
                return np.nan
            else:
                return np.mean(x[-window_size:])
        # Create a new column for the axis with the rolling window data
        combined_data[f"{axis}_window"] = combined_data[axis].rolling(
            window=window_size, min_periods=1).apply(
                rolling_window, raw=True)

        # Remove any rows with NaN values
        combined_data.dropna(inplace=True)

        # Create a new data frame with the rolling window data and their respective timestamps
        rolling_data = pd.DataFrame({
            "timestamp": combined_data["timestamp"],
            f"{axis}_window": combined_data[f"{axis}_window"]
        })

        # Create a new column for the person number
        rolling_data["person"] = person

        # Save the rolling data to a new file
        rolling_data.to_csv(f"rollingData/rollingData_person{person}_{axis}.csv", index=False)