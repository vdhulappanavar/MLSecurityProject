import os
import math
import pandas as pd

# Set the walking threshold
walking_threshold = 2.0

# Create an empty DataFrame to hold the labeled data
data = pd.DataFrame(columns=["timestamp", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z", "label"])

# Loop through each user's data
for user_folder in os.listdir("./data/Training Data"):
    user_folder_path = os.path.join("./data/Training Data", user_folder)
    
    # Loop through each device's data
    for device_folder in os.listdir(user_folder_path):
        device_folder_path = os.path.join(user_folder_path, device_folder)
        print("device_folder, device_folder_path", device_folder, device_folder_path)
        accel_data = ""
        gyro_data = ""
        
        # Load the linear accelerometer data
        if "linearAccelData" in device_folder:
            with open(os.path.join(device_folder_path, "linearAccelData.txt"), "r") as f:
                accel_data = f.readlines()
        
        # Load the gyro data
        if "gyroData" in device_folder:
            with open(os.path.join(device_folder_path, "gyroData.txt"), "r") as f:
                gyro_data = f.readlines()
                
        # Loop through each line of the linear accelerometer data and label it as either "walking" or "climbing"
        for accel_line, gyro_line in zip(accel_data, gyro_data):
            accel_vals = accel_line.strip().split(",")
            gyro_vals = gyro_line.strip().split(",")
            
            # Compute the magnitude of the linear acceleration vector
            accel_mag = math.sqrt(float(accel_vals[1])**2 + float(accel_vals[2])**2 + float(accel_vals[3])**2)
            
            # Determine if the person is walking or climbing based on the magnitude of the linear acceleration vector
            if accel_mag < walking_threshold:
                label = "walking"
            else:
                label = "climbing"
            
            # Append the label to the current line of data
            labeled_line = accel_line.strip() + "," + gyro_line.strip() + "," + label
            
            # Append the labeled data to the DataFrame
            data = data.append({"timestamp": accel_vals[0], "accel_x": accel_vals[1], "accel_y": accel_vals[2], "accel_z": accel_vals[3], "gyro_x": gyro_vals[1], "gyro_y": gyro_vals[2], "gyro_z": gyro_vals[3], "label": label}, ignore_index=True)

# Write the labeled data to a new CSV file
data.to_csv("labeled_data.csv", index=False)
