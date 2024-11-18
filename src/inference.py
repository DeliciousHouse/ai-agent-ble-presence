import pandas as pd
import numpy as np
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_file = "/app/src/models/ble_presence_model.pkl"
encoder_file = "/app/src/models/label_encoder.pkl"

# Define feature columns used during training
feature_columns = (
    'hour', 'day_of_week', 'distance_to_balcony', 'distance_to_office',
    'distance_to_sky_floor', 'distance_to_master_bedroom',
    'distance_to_kitchen', 'distance_to_garage',
    'distance_to_lounge', 'distance_to_master_bathroom'
)

def load_model(file_path):
    """
    Load the trained XGBoost model from file.
    """
    try:
        model = joblib.load(file_path)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def load_label_encoder(file_path):
    """
    Load the LabelEncoder from file.
    """
    try:
        le = joblib.load(file_path)
        logger.info("Label encoder loaded successfully.")
        return le
    except Exception as e:
        logger.error(f"Failed to load label encoder: {e}")
        raise

def prepare_input(device_features):
    """
    Prepare the input data to match the feature columns used during training.
    """
    # Convert device_features dictionary to a DataFrame
    input_df = pd.DataFrame([device_features])

    # Ensure all expected features are present
    input_df = input_df.reindex(columns=feature_columns)

    # Fill missing values with 0 (or another suitable strategy)
    input_df.fillna(0, inplace=True)
    
    return input_df

def predict_room(device_features):
    """
    Predict the room based on input features.
    """
    try:
        # Load the trained model and label encoder
        model = load_model(model_file)
        le = load_label_encoder(encoder_file)

        # Prepare input data
        input_df = prepare_input(device_features)

        # Make the prediction
        encoded_prediction = model.predict(input_df)
        predicted_room = le.inverse_transform([encoded_prediction[0]])[0]  # Decode the label
        logger.info(f"Prediction made successfully: {predicted_room}")
        return predicted_room
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return None
