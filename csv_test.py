import pandas as pd
import os

data_log_file = "test_sensor_data.csv"

def initialize_csv():
    headers = ["timestamp", "device", "estimated_room"]
    if not os.path.isfile(data_log_file):
        pd.DataFrame(columns=headers).to_csv(data_log_file, index=False)
        print(f"CSV Initialized with headers: {headers}")
    else:
        print("CSV already exists.")

initialize_csv()