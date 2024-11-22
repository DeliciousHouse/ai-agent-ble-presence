import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from model_evaluation import evaluate_model

# Define paths for saving models and encoders
model_file = "/app/src/models/ble_presence_model.pkl"
encoder_file = "/app/src/models/label_encoder.pkl"

def train_model(X, y, area_categories):
    """
    Trains the XGBoost model for BLE-based presence detection and saves the model and LabelEncoder.

    Parameters:
        X (pd.DataFrame): Features for training the model.
        y (np.ndarray or pd.Series): Encoded target labels.
        area_categories (np.ndarray): Array of area names corresponding to label encoding.
    """
    # Debugging checks
    print("Training model with the following features:")
    print(X.columns.tolist())
    print(f"Type of X: {type(X)}")  # Should be <class 'pandas.core.frame.DataFrame'>
    print(f"Type of y: {type(y)}")  # Should be <class 'numpy.ndarray'> or <class 'pandas.core.series.Series'>

    # Define all expected areas, including new areas
    all_areas = [
        'lounge', 'master_bedroom', 'kitchen', 'balcony', 'garage', 
        'office', 'front_porch', 'driveway', 'nova_s_room', 'christian_s_room', 
        'sky_floor', 'master_bathroom', 'backyard', 'dining_room', 
        'laundry_room', 'dressing_room',
    ]
    # Initialize the LabelEncoder with the area categories
    le = LabelEncoder()
    le.classes_ = np.array(area_categories)

    # Save the encoder for later use during inference
    joblib.dump(le, encoder_file)
    print(f"Label encoder saved to {encoder_file}.")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define and train the model
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("Model training completed.")

    # Evaluate the model
    evaluate_model(model, X_test, y_test, target_names=le.classes_)
    print("Model evaluation completed.")

    # Save the trained model
    joblib.dump(model, model_file)
    print(f"Model saved to {model_file}.")

    return model