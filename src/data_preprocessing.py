import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

data_log_file = "/app/data/sensor_data.csv"

def load_data():
    """
    Load data from CSV file, handling any bad lines and ensuring timestamp conversion.
    """
    try:
        df = pd.read_csv(data_log_file, on_bad_lines='skip')
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        # Drop rows with invalid timestamps
        df = df.dropna(subset=['timestamp'])
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Data log file not found at {data_log_file}.")
    except pd.errors.ParserError:
        raise ValueError("Error parsing the CSV file.")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")

def preprocess_data(df):
    """
    Preprocess data to extract feature columns and target variable.
    """
    # Remove any _ble suffix from area names if present
    df.columns = [col.replace('_ble', '') for col in df.columns]
    
    # Ensure timestamp is in datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])

    # Extract features: hour and day_of_week
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    # List all possible distance columns based on BLE_DEVICES
    distance_columns = [
        'distance_to_balcony', 'distance_to_office', 
        'distance_to_sky_floor', 'distance_to_master_bedroom', 
        'distance_to_kitchen', 'distance_to_garage', 
        'distance_to_lounge', 'distance_to_master_bathroom'
    ]

    # Add missing distance columns and fill with NaN
    for col in distance_columns:
        if col not in df.columns:
            df[col] = np.nan  # Use 0 if you prefer non-missing values

    # Define feature columns for training
    feature_columns = ['hour', 'day_of_week'] + distance_columns

    # Check if all feature columns exist
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(f"The following feature columns are missing: {missing_features}")

    X = df[feature_columns]  # Features

    # Encode target variable
    if 'estimated_area' not in df.columns:
        raise KeyError("The column 'estimated_area' is missing from the data.")
    df['estimated_area'] = df['estimated_area'].astype(str)  # Ensure target is string
    le = LabelEncoder()
    y = le.fit_transform(df['estimated_area'])

    # For debugging: Inspect the encoded labels
    area_mapping = dict(zip(le.classes_, range(len(le.classes_))))
    print("area Categories and Encoded Values:", area_mapping)

    # Save area categories for later decoding
    area_categories = le.classes_

    # Handle missing values in features
    # X = X.fillna(0)  # Replace with appropriate strategy if needed

    return X, y, area_categories