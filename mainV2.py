import asyncio
import json
import websockets
import requests
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import os
from dotenv import load_dotenv
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import traceback
import logging
import aiofiles
import random
from sklearn.impute import SimpleImputer
import re

# Load environment variables from .env file
load_dotenv()

# Configuration from environment variables
HOME_ASSISTANT_WS_URL = os.getenv("HOME_ASSISTANT_WS_URL")  # e.g., ws://192.168.86.91:8123/api/websocket
HOME_ASSISTANT_WSS_URL = os.getenv("HOME_ASSISTANT_WSS_URL")  # e.g., wss://your-home-assistant-url/api/websocket
HOME_ASSISTANT_API_URL = os.getenv("HOME_ASSISTANT_API_URL")  # e.g., http://192.168.86.91:8123/api
HOME_ASSISTANT_CLOUD_API_URL = os.getenv("HOME_ASSISTANT_CLOUD_API_URL")  # Optional
LONG_LIVED_ACCESS_TOKEN = os.getenv("HA_LONG_LIVED_ACCESS_TOKEN")

if not LONG_LIVED_ACCESS_TOKEN:
    raise ValueError("Home Assistant Long-Lived Access Token not set. Please set HA_LONG_LIVED_ACCESS_TOKEN in your .env file.")

# Headers for REST API calls
HEADERS = {
    "Authorization": f"Bearer {LONG_LIVED_ACCESS_TOKEN}",
    "content-type": "application/json",
}

# Define the device names and areas
DEVICE_NAMES = ["madisons_iphone", "bkam_iphone", "nova_iphone", "bkam_apple_watch"]
MODEL_AREAS = [  # Areas used in model training
    "master_bedroom",
    "master_bathroom",
    "lounge",
    "kitchen",
    "office",
    "sky_floor",
    "balcony"
]
BLE_DEVICES = MODEL_AREAS.copy()  # Assuming BLE_DEVICES are the ones used in the model

# Log file paths
automation_event_log_file = "/app/data/automation_light_events.csv"
manual_event_log_file = "/app/data/manual_light_events.csv"
sensor_data_log_file = "/app/data/sensor_data.csv"
override_log_file = "/app/data/override_log.csv"
location_log_file = "/app/data/location_log.json"
ai_agent_log_file = "/app/data/ai_agent.log"
analysis_output_file = "/app/data/analysis_output.txt"
model_file = "/app/src/models/ble_presence_model.pkl"

# Initialize CSV with headers
def initialize_csv(logger):
    headers = ["timestamp", "device", "estimated_area"] + [
        f"distance_to_{area}" for area in MODEL_AREAS
    ]
    try:
        if not os.path.isfile(sensor_data_log_file):
            pd.DataFrame(columns=headers).to_csv(sensor_data_log_file, index=False)
            logger.info(f"Initialized CSV with headers: {headers}")
        else:
            # Ensure all required columns are present; add missing ones with default values
            df = pd.read_csv(sensor_data_log_file)
            missing_columns = set(headers) - set(df.columns)
            if missing_columns:
                for col in missing_columns:
                    df[col] = np.nan  # Initialize missing columns with NaN
                df.to_csv(sensor_data_log_file, index=False)
                logger.info(f"Added missing columns to CSV: {missing_columns}")
    except Exception as e:
        logger.error(f"Failed to initialize CSV: {e}")

# Define your local timezone
LOCAL_TIMEZONE = ZoneInfo("America/Los_Angeles")  # Replace with your local timezone

# Function to get the current local time
def get_local_time():
    return datetime.now(LOCAL_TIMEZONE)

# Shared sensor data and its lock
sensor_data = {}
sensor_data_lock = asyncio.Lock()

# Additional Shared States
current_areas = {}               # Maps device to current area
last_action_time = {}           # Maps device to last action timestamp
min_time_between_actions = 5     # Minimum seconds between actions per device
override_timers = {}            # Maps area to override information
override_duration = 300         # Duration (in seconds) for which an override is active

# Function to fetch user name based on user_id
def fetch_user_name(logger, user_id):
    """
    Fetches the user's name from Home Assistant using the user_id.
    """
    url = f"{HOME_ASSISTANT_API_URL}/users/{user_id}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            user_data = response.json()
            return user_data.get("name", "unknown")
        else:
            logger.error(f"Failed to fetch user name for user_id {user_id}: {response.text}")
            return "unknown"
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching user name for user_id {user_id}: {e}")
        return "unknown"

# Log sensor data to the CSV
def log_sensor_data(logger, device, estimated_area, distances):
    current_time = get_local_time()
    timestamp = current_time.isoformat()
    data_entry = {"timestamp": timestamp, "device": device, "estimated_area": estimated_area}
    for area, distance in distances.items():
        data_entry[f"distance_to_{area}"] = distance  # Ensure correct key naming
    try:
        df = pd.DataFrame([data_entry])
        if not os.path.isfile(sensor_data_log_file):
            df.to_csv(sensor_data_log_file, index=False)
        else:
            df.to_csv(sensor_data_log_file, mode='a', header=False, index=False)
        logger.debug(f"Logged sensor data to {sensor_data_log_file}: {data_entry}")
    except Exception as e:
        logger.error(f"Failed to log sensor data: {e}")

# Function to call Home Assistant services with dynamic brightness and color_temp
async def call_service(logger, domain, service, service_data):
    for api_url in [HOME_ASSISTANT_API_URL, HOME_ASSISTANT_CLOUD_API_URL]:
        if not api_url:
            continue
        url = f"{api_url}/services/{domain}/{service}"
        try:
            logger.debug(f"Attempting to call service {domain}.{service} at {url} with data {service_data}")
            response = requests.post(url, headers=HEADERS, json=service_data, timeout=10)
            if response.status_code == 200:
                logger.info(f"Called service {domain}.{service} with {service_data} using {api_url}")
                return True
            else:
                logger.error(f"Failed to call service {domain}.{service} using {api_url}: {response.text}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling service {domain}.{service} using {api_url}: {e}")
    logger.error(f"All attempts to call service {domain}.{service} failed.")
    return False

# Function to log automation actions with latest sensor data
async def log_automation_action(logger, entity_id, action, brightness=None, user="unknown"):
    current_time = get_local_time()
    timestamp = current_time.isoformat()
    hour = current_time.hour
    log_entry = {
        "timestamp": timestamp,
        "entity_id": entity_id,
        "action": action,
        "brightness": brightness,
        "hour": hour,
        "user": user
    }

    # Append log entry to the CSV file
    try:
        df = pd.DataFrame([log_entry])
        if not os.path.isfile(automation_event_log_file):
            df.to_csv(automation_event_log_file, index=False)
        else:
            df.to_csv(automation_event_log_file, mode='a', header=False, index=False)
        logger.info(f"Logged automation action: {log_entry}")
    except Exception as e:
        logger.error(f"Failed to log automation action: {e}")

# Function to log manual light events with latest sensor data
async def log_manual_light_event(logger, entity_id, action, brightness=None, user="unknown"):
    current_time = get_local_time()
    timestamp = current_time.isoformat()
    hour = current_time.hour
    log_entry = {
        "timestamp": timestamp,
        "entity_id": entity_id,
        "action": action,
        "brightness": brightness,
        "hour": hour,
        "user": user
    }

    try:
        df = pd.DataFrame([log_entry])
        if not os.path.isfile(manual_event_log_file):
            df.to_csv(manual_event_log_file, index=False)
        else:
            df.to_csv(manual_event_log_file, mode='a', header=False, index=False)
        logger.info(f"Logged manual light event: {log_entry}")
    except Exception as e:
        logger.error(f"Failed to log manual light event: {e}")

# Function to log location
async def log_location(logger, device, area, action, user="unknown"):
    current_time = get_local_time()
    timestamp = current_time.isoformat()
    log_entry = {"timestamp": timestamp, "device": device, "area": area, "action": action, "user": user}
    log_dir = "/app/data"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "location_log.json")
    try:
        async with aiofiles.open(log_file, "a") as logfile:
            await logfile.write(json.dumps(log_entry) + "\n")
        logger.info(f"Logged location: {log_entry}")
    except Exception as e:
        logger.error(f"Failed to write log entry: {e}")

# Function to determine brightness and color temperature based on local time
def get_dynamic_brightness_and_temperature():
    current_hour = get_local_time().hour

    # Define brightness and temperature based on time
    if 6 <= current_hour < 12:  # Morning
        brightness = 200  # Adjust as needed
        color_temp = 4000  # Cool white for morning
    elif 12 <= current_hour < 18:  # Afternoon
        brightness = 255  # Maximum brightness
        color_temp = 3500  # Natural white for afternoon
    elif 18 <= current_hour < 22:  # Evening
        brightness = 150  # Reduced brightness
        color_temp = 3000  # Warm white for evening
    else:  # Nighttime
        brightness = 50   # Low brightness
        color_temp = 2700  # Very warm white for nighttime

    return brightness, color_temp

# Function to dynamically generate AREA_ACTIONS based on sensor_data
def get_area_actions():
    brightness, color_temp = get_dynamic_brightness_and_temperature()
    current_hour = get_local_time().hour
    is_daytime = 6 <= current_hour < 18  # Define daytime from 6 AM to 6 PM

    area_actions = {}
    for area in MODEL_AREAS:
        if is_daytime and area not in ["master_bedroom", "master_bathroom", "office", "sky_floor", "garage"]:
            # During daytime, turn off lights except specified areas
            actions = {
                "enter": [
                    (
                        "light",
                        "turn_off",
                        {
                            "entity_id": f"light.{area}_lights"
                        }
                    )
                ],
                "exit": []  # No action on exit during daytime for these areas
            }
        else:
            # Normal operation
            actions = {
                "enter": [
                    (
                        "light",
                        "turn_on",
                        {
                            "entity_id": f"light.{area}_lights",
                            "brightness": brightness,
                            "color_temp": color_temp
                        }
                    )
                ],
                "exit": [
                    (
                        "light",
                        "turn_off",
                        {
                            "entity_id": f"light.{area}_lights"
                        }
                    )
                ]
            }
        area_actions[area] = actions
    return area_actions

# Function to extract device name and area from entity_id
def extract_device_and_area_from_entity_id(entity_id):
    """
    Extracts the device name and area from the entity_id.
    Assumes entity_id follows the pattern: 'sensor.device_name_ble_distance_to_model_area'
    Returns (device_name, model_area) or (None, None) if pattern doesn't match.
    """
    pattern = r"sensor\.(?P<device_name>.+)_ble_distance_to_(?P<model_area>.+)"
    match = re.match(pattern, entity_id)
    if match:
        device_name = match.group("device_name")
        model_area = match.group("model_area")
        return device_name, model_area
    return None, None

# Function to prepare features for prediction
def prepare_features(sensor_data, device, areas):
    # Initialize a dictionary for features
    features = {}

    # Add distance features based on MODEL_AREAS
    for area in areas:
        key = f"distance_to_{area}"
        # Extract all relevant sensors for this area based on naming pattern
        relevant_sensors = [
            sensor_id for sensor_id in sensor_data.keys()
            if sensor_id.startswith(f"sensor.{device}_ble_distance_to_{area}")
        ]
        distances = []
        for sensor in relevant_sensors:
            distance_value = sensor_data.get(sensor, np.nan)
            try:
                distance = float(distance_value) if not pd.isna(distance_value) else np.nan
                distances.append(distance)
            except (ValueError, TypeError):
                continue  # Skip invalid entries
        if distances:
            # For presence detection, using the minimum distance makes sense
            aggregated_distance = min(distances)
        else:
            aggregated_distance = np.nan
        features[key] = aggregated_distance

    # Add time-based features
    current_time = get_local_time()
    features['hour'] = current_time.hour
    features['day_of_week'] = current_time.weekday()

    # Add device ID as categorical
    device_id = device
    # Map device to an index (assuming device_categories are handled during model training)
    device_category_map = {device: idx for idx, device in enumerate(DEVICE_NAMES)}
    device_idx = device_category_map.get(device_id, -1)
    features['device_id'] = device_idx

    # Convert features to a DataFrame
    feature_df = pd.DataFrame([features])

    # Handle missing values (NaN) by imputing with median (consistent with training)
    imputer = SimpleImputer(strategy='median')
    feature_df_imputed = imputer.fit_transform(feature_df)

    return feature_df_imputed  # Returns a numpy array suitable for prediction

# Define floor and section BLE data
FLOOR_MAP = {
    "first_floor": {
        "front": ["front_porch", "master_bedroom", "nova_s_room", "bathroom"],
        "middle": ["master_bathroom", "dining_room", "lounge", "driveway"],
        "back": ["kitchen", "backyard", "garage"]
    },
    "second_floor": {
        "front": ["dressing_room"],
        "middle": ["office", "sky_floor"],
        "back": ["balcony"]
    }
}

def determine_floor_and_section(area):
    for floor, sections in FLOOR_MAP.items():
        for section, areas_in_section in sections.items():
            if area in areas_in_section:
                return floor, section
    return None, None

# Function to check if an area is currently overridden
def is_area_overridden(area):
    # Check override_timers dictionary
    current_time = time.time()
    override_info = override_timers.get(area)
    return override_info is not None and override_info['end_time'] > current_time

# Function to handle area changes
async def handle_area_change(logger, device, area, action_type, AREA_ACTIONS, user="unknown"):
    current_time = time.time()
    logger.debug(f"Handling area change for device {device}: {action_type} {area} at {current_time}")
    if last_action_time.get(device, 0) + min_time_between_actions > current_time:
        logger.info(f"Skipping action for {device} {action_type}ing {area} due to debounce.")
        return
    # Access shared sensor_data
    async with sensor_data_lock:
        current_sensor_data = sensor_data.copy()
    brightness, color_temp = get_dynamic_brightness_and_temperature()

    logger.info(f"{device} {action_type}ing area: {area}")
    actions = AREA_ACTIONS.get(area, {}).get(action_type, [])
    current_floor, current_section = determine_floor_and_section(area)
    prev_area = current_areas.get(device)
    if prev_area:
        prev_floor, prev_section = determine_floor_and_section(prev_area)
        if current_floor != prev_floor:
            logger.warning(f"Invalid transition detected: {prev_area} (floor {prev_floor}) to {area} (floor {current_floor}). Skipping.")
            return 
        elif current_section != prev_section:
            logger.info(f"Transition within the same floor: {prev_area} to {area}. Proceeding.") 

    if is_area_overridden(area):
        logger.info(f"Area {area} is currently overridden. Skipping {action_type} actions.")
        return

    for domain, service, service_data in actions:
        # Dynamically include brightness and color_temp if action is 'turn_on'
        if service == "turn_on":
            service_data["brightness"] = brightness
            service_data["color_temp"] = color_temp
        logger.debug(f"Calling service {domain}.{service} with data {service_data}")
        result = await call_service(logger, domain, service, service_data)
        if result:
            logger.info(f"Service call successful for {domain}.{service} in {area}.")
        else:
            logger.warning(f"Service call failed for {domain}.{service} in {area}. Implementing fallback...")

    await log_location(logger, device, area, action_type, user)
    last_action_time[device] = current_time
    current_areas[device] = area

# Function to log override events
async def log_override_event(logger, area, light_entity_id, user="unknown"):
    current_time = get_local_time()
    timestamp = current_time.isoformat()
    hour = current_time.hour
    day_of_week = current_time.weekday()
    override_entry = {
        "timestamp": timestamp,
        "area": area,
        "light_entity_id": light_entity_id,
        "user": user,
        "hour": hour,
        "day_of_week": day_of_week,
    }
    try:
        df = pd.DataFrame([override_entry])
        if not os.path.isfile(override_log_file):
            df.to_csv(override_log_file, index=False)
        else:
            df.to_csv(override_log_file, mode='a', header=False, index=False)
        logger.info(f"Logged override event: {override_entry}")
    except Exception as e:
        logger.error(f"Failed to log override event: {e}")

# Function to set override
def set_override(area, logger):
    override_end_time = time.time() + override_duration
    override_timers[area] = {'end_time': override_end_time}
    # Schedule the reset_override coroutine
    loop = asyncio.get_event_loop()
    loop.create_task(reset_override(logger, area))

# Function to reset override
async def reset_override(logger, area):
    await asyncio.sleep(override_duration)
    override_timers.pop(area, None)
    logger.info(f"Override for area {area} has been reset.")

# Function to cleanup old data
async def cleanup_old_data(logger, retention_days=30):
    for file in [sensor_data_log_file, override_log_file]:
        if os.path.isfile(file):
            try:
                # Read CSV with error handling for bad lines
                df = pd.read_csv(file, parse_dates=['timestamp'], on_bad_lines='skip')
                cutoff_date = get_local_time() - timedelta(days=retention_days)
                df = df[df['timestamp'] >= cutoff_date]
                df.to_csv(file, index=False)
                logger.info(f"Cleaned up data in {file} older than {retention_days} days.")
            except Exception as e:
                logger.error(f"Failed to clean up old data in {file}: {e}")

# Function to analyze data
async def analyze_data(logger):
    logger.info("Starting advanced data analysis coroutine")
    retention_days = 30
    while True:
        try:
            await cleanup_old_data(logger, retention_days=retention_days)
            if os.path.isfile(sensor_data_log_file):
                try:
                    # Read CSV with error handling for bad lines
                    df = pd.read_csv(sensor_data_log_file, parse_dates=['timestamp'], on_bad_lines='skip')
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

                    # Proceed with data analysis...
                    df.dropna(inplace=True)
                    df['hour'] = df['timestamp'].dt.hour
                    df['day_of_week'] = df['timestamp'].dt.dayofweek

                    # Encode categorical variables
                    df['device_id'] = df['device'].astype('category').cat.codes
                    device_categories = df['device'].astype('category').cat.categories.tolist()

                    # Encode areas
                    df['area_id'] = df['estimated_area'].apply(lambda x: MODEL_AREAS.index(x) if x in MODEL_AREAS else -1)
                    area_categories_list = MODEL_AREAS  # Avoid overwriting the global area_categories

                    # Feature columns
                    distance_columns = [f"distance_to_{area}" for area in MODEL_AREAS]
                    distance_columns.sort()

                    X = df[distance_columns + ['hour', 'day_of_week', 'device_id']]
                    y = df['area_id']

                    valid_indices = y != -1
                    X = X[valid_indices]
                    y = y[valid_indices]

                    if X.empty:
                        logger.warning("No valid data available for training after filtering. Skipping model training.")
                    else:
                        feature_names = X.columns.tolist()

                        # Handle missing values with imputation
                        imputer = SimpleImputer(strategy='median')
                        X_imputed = imputer.fit_transform(X)

                        # Train the model
                        model = XGBClassifier(
                            n_estimators=100,
                            max_depth=6,
                            learning_rate=0.1,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            random_state=42,
                            use_label_encoder=False,
                            eval_metric='mlogloss'
                        )
                        model.fit(X_imputed, y)

                        y_pred = model.predict(X_imputed)
                        report = classification_report(y, y_pred, target_names=area_categories_list)

                        # Save the model
                        joblib.dump(model, model_file)

                        # Write analysis report
                        async with aiofiles.open(analysis_output_file, 'w') as f:
                            await f.write("Classification Report:\n")
                            await f.write(report)
                        logger.info(f"Advanced analysis results written to {analysis_output_file}")
                except Exception as e:
                    logger.error(f"Error during data analysis: {e}")
            else:
                logger.info("No data available for analysis")
        except Exception as e:
            logger.error(f"Error during advanced data analysis: {e}")
            logger.debug(traceback.format_exc())
        await asyncio.sleep(3600)  # Run analysis every hour

# Function to monitor overrides
async def monitor_overrides(logger):
    ws_urls = [HOME_ASSISTANT_WS_URL, HOME_ASSISTANT_WSS_URL]
    backoff_base = 1  # Start with 1 second
    backoff_max = 60  # Maximum backoff time
    while True:
        for ws_url in ws_urls:
            if not ws_url:
                continue
            try:
                async with websockets.connect(ws_url, timeout=10) as websocket:
                    logger.info(f"Connected to {ws_url} for override monitoring")

                    # Authenticate
                    await websocket.send(json.dumps({"type": "auth", "access_token": LONG_LIVED_ACCESS_TOKEN}))
                    auth_response = await websocket.recv()
                    auth_result = json.loads(auth_response)
                    if auth_result.get("type") != "auth_ok":
                        logger.error(f"Authentication failed with {ws_url}: {auth_result.get('message', '')}")
                        continue
                    logger.info(f"Authenticated successfully with {ws_url} for override monitoring")

                    # Subscribe to state_changed events
                    await websocket.send(json.dumps({"id": 200, "type": "subscribe_events", "event_type": "state_changed"}))
                    logger.debug("Subscribed to state_changed events for override monitoring")

                    backoff = backoff_base  # Reset backoff after successful connection

                    while True:
                        try:
                            message = await websocket.recv()
                            event = json.loads(message)
                            if event.get("type") == "event" and event["event"].get("event_type") == "state_changed":
                                entity_id = event["event"]["data"]["entity_id"]
                                new_state = event["event"]["data"]["new_state"]["state"]
                                old_state = event["event"]["data"]["old_state"]["state"]
                                context = event["event"].get("context", {})
                                user_id = context.get("user_id", "unknown")
                                user_name = "unknown"

                                if user_id != "unknown":
                                    user_name = fetch_user_name(logger, user_id)

                                # Extract device and area from entity_id
                                device_name, area = extract_device_and_area_from_entity_id(entity_id)
                                if device_name and area in MODEL_AREAS:
                                    light_entity_id = f"light.{area}_lights"
                                    if entity_id == light_entity_id:
                                        if old_state == "on" and new_state == "off":
                                            logger.info(f"Manual override detected for {light_entity_id}")
                                            await log_override_event(logger, area=area, light_entity_id=light_entity_id, user=user_name)
                                            set_override(area, logger)
                                        elif old_state == "off" and new_state == "on":
                                            logger.info(f"Manual override turned on for {light_entity_id}")
                                            await log_override_event(logger, area=area, light_entity_id=light_entity_id, user=user_name)
                                            set_override(area, logger)
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error in monitor_overrides: {e}")
                            continue
                        except websockets.exceptions.ConnectionClosedError as e:
                            logger.error(f"WebSocket connection closed in monitor_overrides: {e}")
                            break  # Break inner loop to attempt reconnection
                        except Exception as e:
                            logger.error(f"Unexpected error in monitor_overrides: {e}")
                            logger.debug(traceback.format_exc())
                            break  # Break inner loop to attempt reconnection
            except Exception as e:
                logger.error(f"Error connecting to {ws_url} for override monitoring: {e}")
                logger.debug(traceback.format_exc())
                # Implement exponential backoff with jitter
                sleep_time = min(backoff_max, backoff_base * 2)
                jitter = random.uniform(0, 1)
                await asyncio.sleep(sleep_time + jitter)
                backoff_base = sleep_time  # Increase backoff
        await asyncio.sleep(5)  # Short delay before next connection attempt

# Function to monitor manual light events
async def monitor_light_events(logger):
    ws_urls = [HOME_ASSISTANT_WS_URL, HOME_ASSISTANT_WSS_URL]
    backoff_base = 1  # Start with 1 second
    backoff_max = 60  # Maximum backoff time
    while True:
        for ws_url in ws_urls:
            if not ws_url:
                continue
            try:
                async with websockets.connect(ws_url, timeout=10) as websocket:
                    logger.info(f"Connected to {ws_url} for manual light event monitoring")

                    # Authenticate
                    await websocket.send(json.dumps({"type": "auth", "access_token": LONG_LIVED_ACCESS_TOKEN}))
                    auth_response = await websocket.recv()
                    auth_result = json.loads(auth_response)
                    if auth_result.get("type") != "auth_ok":
                        logger.error(f"Authentication failed with {ws_url}: {auth_result.get('message', '')}")
                        continue
                    logger.info(f"Authenticated successfully with {ws_url} for manual light events")

                    # Subscribe to state_changed events
                    await websocket.send(json.dumps({"id": 101, "type": "subscribe_events", "event_type": "state_changed"}))
                    logger.debug("Subscribed to state_changed events for manual light event monitoring")

                    backoff = backoff_base  # Reset backoff after successful connection

                    while True:
                        try:
                            message = await websocket.recv()
                            event = json.loads(message)
                            if event.get("type") == "event" and event["event"].get("event_type") == "state_changed":
                                entity_id = event["event"]["data"]["entity_id"]
                                new_state = event["event"]["data"]["new_state"]
                                old_state = event["event"]["data"]["old_state"]
                                context = event["event"].get("context", {})
                                user_id = context.get("user_id", "unknown")
                                user_name = "unknown"

                                if user_id != "unknown":
                                    user_name = fetch_user_name(logger, user_id)

                                if entity_id.startswith("light."):
                                    if new_state and old_state:
                                        if new_state["state"] != old_state["state"]:
                                            action = "turned_on" if new_state["state"] == "on" else "turned_off"
                                            logger.info(f"Manual light event: {entity_id} has been {action} by {user_name}")
                                            await log_manual_light_event(logger, entity_id, action, user=user_name)
                                        # Check for brightness change
                                        if "attributes" in new_state and "brightness" in new_state["attributes"]:
                                            new_brightness = new_state["attributes"]["brightness"]
                                            old_brightness = old_state["attributes"].get("brightness", np.nan)
                                            if not pd.isna(new_brightness) and not pd.isna(old_brightness) and new_brightness != old_brightness:
                                                action = "brightness_changed"
                                                brightness = new_brightness
                                                logger.info(f"Manual light event: {entity_id} brightness changed to {brightness} by {user_name}")
                                                await log_manual_light_event(logger, entity_id, action, brightness, user=user_name)
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error in monitor_light_events: {e}")
                            continue
                        except websockets.exceptions.ConnectionClosedError as e:
                            logger.error(f"WebSocket connection closed in monitor_light_events: {e}")
                            break  # Break inner loop to attempt reconnection
                        except Exception as e:
                            logger.error(f"Unexpected error in monitor_light_events: {e}")
                            logger.debug(traceback.format_exc())
                            break  # Break inner loop to attempt reconnection
            except Exception as e:
                logger.error(f"Error connecting to {ws_url} for manual light events: {e}")
                logger.debug(traceback.format_exc())
                # Implement exponential backoff with jitter
                sleep_time = min(backoff_max, backoff_base * 2)
                jitter = random.uniform(0, 1)
                await asyncio.sleep(sleep_time + jitter)
                backoff_base = sleep_time  # Increase backoff
        await asyncio.sleep(5)  # Short delay before next connection attempt

# Function to monitor automation events
async def monitor_automation_events(logger):
    ws_urls = [HOME_ASSISTANT_WS_URL, HOME_ASSISTANT_WSS_URL]
    backoff_base = 1  # Start with 1 second
    backoff_max = 60  # Maximum backoff time
    while True:
        for ws_url in ws_urls:
            if not ws_url:
                continue
            try:
                async with websockets.connect(ws_url, timeout=10) as websocket:
                    logger.info(f"Connected to {ws_url} for automation event monitoring")

                    # Authenticate
                    await websocket.send(json.dumps({"type": "auth", "access_token": LONG_LIVED_ACCESS_TOKEN}))
                    auth_response = await websocket.recv()
                    auth_result = json.loads(auth_response)
                    if auth_result.get("type") != "auth_ok":
                        logger.error(f"Authentication failed with {ws_url}: {auth_result.get('message', '')}")
                        continue
                    logger.info(f"Authenticated successfully with {ws_url} for automation event monitoring")

                    # Subscribe to state_changed events
                    await websocket.send(json.dumps({"id": 200, "type": "subscribe_events", "event_type": "state_changed"}))
                    logger.debug("Subscribed to state_changed events for automation event monitoring")

                    backoff = backoff_base  # Reset backoff after successful connection

                    while True:
                        try:
                            message = await websocket.recv()
                            event = json.loads(message)
                            if event.get("type") == "event" and event["event"].get("event_type") == "state_changed":
                                entity_id = event["event"]["data"]["entity_id"]
                                automation_state = event["event"]["data"]["new_state"]["state"]
                                context = event["event"].get("context", {})
                                user_id = context.get("user_id", "unknown")
                                user_name = "unknown"

                                if user_id != "unknown":
                                    user_name = fetch_user_name(logger, user_id)

                                if entity_id.startswith("automation.") and automation_state == "on":
                                    # Derive affected light entity
                                    affected_light = entity_id.replace("automation.", "light.")
                                    brightness, color_temp = get_dynamic_brightness_and_temperature()
                                    logger.info(f"Automation triggered: {automation_state} for {affected_light}")
                                    await log_automation_action(logger, affected_light, "triggered", brightness=brightness, user=user_name)
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error in monitor_automation_events: {e}")
                            continue
                        except websockets.exceptions.ConnectionClosedError as e:
                            logger.error(f"WebSocket connection closed in monitor_automation_events: {e}")
                            break  # Break inner loop to attempt reconnection
                        except Exception as e:
                            logger.error(f"Unexpected error in monitor_automation_events: {e}")
                            logger.debug(traceback.format_exc())
                            break  # Break inner loop to attempt reconnection
            except Exception as e:
                logger.error(f"Error connecting to {ws_url} for automation event monitoring: {e}")
                logger.debug(traceback.format_exc())
                # Implement exponential backoff with jitter
                sleep_time = min(backoff_max, backoff_base * 2)
                jitter = random.uniform(0, 1)
                await asyncio.sleep(sleep_time + jitter)
                backoff_base = sleep_time  # Increase backoff
        await asyncio.sleep(5)  # Short delay before next connection attempt

# Function to update sensor data periodically
async def sensor_data_updater(logger):
    """
    Periodically fetches sensor data from Home Assistant and updates the shared sensor_data dictionary.
    """
    while True:
        try:
            logger.debug("Fetching sensor data from Home Assistant.")
            url = f"{HOME_ASSISTANT_API_URL}/states"
            response = requests.get(url, headers=HEADERS, timeout=10)
            if response.status_code == 200:
                states = response.json()
                async with sensor_data_lock:
                    for state in states:
                        entity_id = state['entity_id']
                        state_value = state['state']
                        # Convert '0' to np.nan
                        if state_value == "0":
                            sensor_data[entity_id] = np.nan
                        else:
                            try:
                                sensor_data[entity_id] = float(state_value)
                            except ValueError:
                                sensor_data[entity_id] = np.nan
                logger.debug("Sensor data updated successfully.")
            else:
                logger.error(f"Failed to fetch sensor states: {response.text}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching sensor states: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in sensor_data_updater: {e}")
            logger.debug(traceback.format_exc())
        await asyncio.sleep(10)  # Update every 10 seconds

# Function to handle inference and actions based on predictions
async def inference_and_handle(logger, model):
    """
    Periodically performs inference for all devices and handles area changes.
    """
    while True:
        try:
            async with sensor_data_lock:
                devices = DEVICE_NAMES.copy()
            AREA_ACTIONS = get_area_actions()
            logger.debug(f"Starting inference for devices: {devices}")
            for device in devices:
                logger.debug(f"Performing inference for device: {device}")
                area = await perform_inference(logger, device, model)
                if area:
                    logger.debug(f"Device {device} inferred to be in area {area}")
                    await handle_area_change(logger, device, area, "enter", AREA_ACTIONS, user="inference")
                else:
                    logger.debug(f"No confident area prediction for device {device}")
            await asyncio.sleep(60)  # Perform inference every 60 seconds
        except Exception as e:
            logger.error(f"Error in inference_and_handle: {e}")
            logger.debug(traceback.format_exc())
            await asyncio.sleep(60)

# Function to perform inference using the trained model
async def perform_inference(logger, device, model):
    """
    Uses the trained model to predict the area of the device based on current sensor data.
    Returns the predicted area if confidence is high enough, else None.
    """
    try:
        if not os.path.isfile(model_file):
            logger.error(f"Model file {model_file} not found. Inference cannot be performed.")
            return None
        # Load the model
        logger.debug(f"Loading model from {model_file} for device {device}")
        # Model is already loaded and passed, no need to load again
        # Access shared sensor_data
        async with sensor_data_lock:
            current_sensor_data = sensor_data.copy()
        # Prepare features
        features = prepare_features(current_sensor_data, device, MODEL_AREAS)
        logger.debug(f"Prepared features for device {device}: {features}")
        if np.isnan(features).any():
            logger.warning(f"Inference: Missing features for device {device}. Skipping prediction.")
            return None
        # Perform prediction
        prediction = model.predict(features)[0]
        logger.debug(f"Model prediction for device {device}: {prediction}")
        # Optionally, get prediction probability
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            confidence = max(probabilities)
            logger.debug(f"Prediction probabilities for device {device}: {probabilities}")
        else:
            confidence = 1.0  # If model doesn't support predict_proba
        predicted_area = MODEL_AREAS[prediction] if 0 <= prediction < len(MODEL_AREAS) else None
        if predicted_area and confidence > 0.6:  # Threshold can be adjusted
            logger.info(f"Inference: Device {device} is in area {predicted_area} with confidence {confidence:.2f}")
            return predicted_area
        else:
            logger.warning(f"Inference: Low confidence ({confidence:.2f}) for device {device}. Predicted area: {predicted_area}")
            return None
    except Exception as e:
        logger.error(f"Error during inference for device {device}: {e}")
        logger.debug(traceback.format_exc())
        return None

# Function to ensure critical area lighting based on presence
async def ensure_critical_area_lighting(logger):
    while True:
        try:
            # Determine if it's dark based on sun state
            sun_response = requests.get(f"{HOME_ASSISTANT_API_URL}/sun/state", headers=HEADERS, timeout=10)
            if sun_response.status_code == 200:
                sun_data = sun_response.json()
                is_dark = not sun_data.get("state", "above_horizon") == "above_horizon"
                logger.debug(f"Sun state: {'dark' if is_dark else 'light'}")
            else:
                logger.warning(f"Failed to fetch sun state: {sun_response.text}")
                is_dark = False  # Default to not dark if unable to fetch

            try:
                for area, presence_sensor in [("balcony", "sensor.balcony_presence"), ("backyard", "sensor.backyard_presence")]:
                    presence_state = sensor_data.get(presence_sensor, "off")
                    light_entity = f"light.{area}_lights"

                    if is_dark and presence_state == "on":
                        # Fetch latest brightness and color_temp based on local time
                        brightness, color_temp = get_dynamic_brightness_and_temperature()
                        service_data = {
                            "entity_id": light_entity,
                            "brightness": brightness,
                            "color_temp": color_temp
                        }
                        logger.debug(f"Ensuring {light_entity} is on due to presence and dark conditions.")
                        result = await call_service(logger, "light", "turn_on", service_data)
                        if result:
                            logger.info(f"Ensured {light_entity} is on due to presence and dark conditions")
                        else:
                            logger.warning(f"Failed to ensure {light_entity} is on. Retrying or implementing fallback...")
            except Exception as e:
                logger.error(f"Error in ensure_critical_area_lighting: {e}")
                logger.debug(traceback.format_exc())

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching sun state: {e}")
            is_dark = False  # Default to not dark if unable to fetch
        except Exception as e:
            logger.error(f"Unexpected error in ensure_critical_area_lighting: {e}")
            logger.debug(traceback.format_exc())

        await asyncio.sleep(60)  # Check every 60 seconds

# Main application logic
async def main():
    # Initialize asynchronous logger
    logger = logging.getLogger('ai_agent')
    logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs during testing

    # Create and add file handler
    file_handler = logging.FileHandler(ai_agent_log_file)
    file_handler.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Create and add stream handler (logs to STDOUT)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs
    stream_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    # Initialize CSV file
    initialize_csv(logger)

    # Load and preprocess data
    try:
        logger.info("Loading and preprocessing data...")
        # Load data without returning a DataFrame
        await asyncio.to_thread(load_data, logger)

        # For training, read and preprocess the data in a separate thread
        # This ensures that load_data has already updated sensor_data
        if not os.path.isfile(sensor_data_log_file):
            logger.error(f"Sensor data file {sensor_data_log_file} not found after loading data. Exiting.")
            return

        df = pd.read_csv(sensor_data_log_file)
        X, y, area_categories = await asyncio.to_thread(preprocess_data, df)
        if X is None or y is None or area_categories is None:
            logger.error("Preprocessing failed. Exiting.")
            return

        # Train the model
        logger.info("Training the model...")
        model = await asyncio.to_thread(train_model, X, y, area_categories)
        if model is None:
            logger.error("Model training failed. Exiting.")
            return

        logger.info("Model training completed.")
    except Exception as e:
        logger.error(f"Error during data loading and model training: {e}")
        logger.debug(traceback.format_exc())
        return

    # Load the trained model once and pass it to inference tasks
    try:
        if not os.path.isfile(model_file):
            logger.error(f"Trained model file {model_file} not found. Exiting.")
            return
        logger.debug(f"Loading trained model from {model_file} for inference.")
        model = joblib.load(model_file)
    except Exception as e:
        logger.error(f"Failed to load the trained model: {e}")
        logger.debug(traceback.format_exc())
        return

    try:
        # Start monitoring tasks
        asyncio.create_task(monitor_overrides(logger))
        asyncio.create_task(ensure_critical_area_lighting(logger))
        asyncio.create_task(monitor_automation_events(logger))
        asyncio.create_task(monitor_light_events(logger))
        asyncio.create_task(sensor_data_updater(logger))  # Ensure sensor_data_updater is running
        asyncio.create_task(inference_and_handle(logger, model))  # Start inference loop with the loaded model
        asyncio.create_task(analyze_data(logger))  # Start data analysis loop

        logger.info("AI agent is running...")
    except Exception as e:
        logger.error(f"Error starting AI agent tasks: {e}")
        logger.debug(traceback.format_exc())
        return

    # Keep the main coroutine running
    while True:
        await asyncio.sleep(3600)  # Sleep for an hour, adjust as needed

# Function to load data (Modified to not return a pandas DataFrame)
def load_data(logger):
    """
    Loads raw sensor data from CSV files and updates the shared sensor_data dictionary.
    Does not return a pandas DataFrame.
    """
    try:
        # Check if the sensor data CSV exists
        if not os.path.isfile(sensor_data_log_file):
            logger.warning(f"Sensor data file {sensor_data_log_file} does not exist. Skipping data load.")
            return

        # Read the sensor data CSV
        df = pd.read_csv(sensor_data_log_file)

        # Iterate over each row and update sensor_data
        for index, row in df.iterrows():
            device = row['device']
            estimated_area = row['estimated_area']
            distances = {f"distance_to_{area}": row.get(f"distance_to_{area}", np.nan) for area in MODEL_AREAS}

            # Log sensor data
            log_sensor_data(logger, device, estimated_area, distances)

        logger.info("Sensor data loaded and shared data updated.")
    except Exception as e:
        logger.error(f"Failed to load and process data: {e}")

# Function to preprocess data (Completed try-except)
def preprocess_data(df):
    """
    Preprocesses the raw sensor data.
    Returns feature matrix X, target vector y, and list of area categories.
    """
    logger = logging.getLogger('ai_agent')
    try:
        # 1. Convert 'timestamp' to datetime if not already
        if 'timestamp' not in df.columns:
            logger.error("'timestamp' column missing from sensor data.")
            return None, None, None

        if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            logger.debug("Converted 'timestamp' to datetime.")
        
        # 2. Drop rows where 'timestamp' conversion failed
        initial_row_count = len(df)
        df.dropna(subset=['timestamp'], inplace=True)
        dropped_rows = initial_row_count - len(df)
        if dropped_rows > 0:
            logger.warning(f"Dropped {dropped_rows} rows due to invalid 'timestamp'.")

        # 3. Extract 'hour' and 'day_of_week'
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        logger.debug("Extracted 'hour' and 'day_of_week' from 'timestamp'.")

        # 4. Encode categorical variables
        if 'device' not in df.columns or 'estimated_area' not in df.columns:
            logger.error("'device' or 'estimated_area' columns missing from sensor data.")
            return None, None, None

        df['device'] = df['device'].astype('category')
        df['estimated_area'] = df['estimated_area'].astype('category')
        area_categories = df['estimated_area'].cat.categories.tolist()
        logger.debug("Encoded 'device' and 'estimated_area' as categorical.")

        # 5. Encode 'area_id'
        df['area_id'] = df['estimated_area'].cat.codes
        logger.debug("Encoded 'area_id' based on 'estimated_area'.")

        # 6. Encode 'device_id'
        device_category_map = {device: idx for idx, device in enumerate(DEVICE_NAMES)}
        df['device_id'] = df['device'].map(device_category_map).fillna(-1).astype(int)
        logger.debug("Encoded 'device_id' based on 'device'.")

        # 7. Ensure all 'distance_to_<area>' columns are present
        for area in MODEL_AREAS:
            col = f"distance_to_{area}"
            if col not in df.columns:
                df[col] = np.nan  # Assign NaN for missing columns
                logger.warning(f"Added missing column '{col}' with NaN.")
        
        # 8. Replace 0s with NaN in 'distance_to_<area>' columns
        for area in MODEL_AREAS:
            col = f"distance_to_{area}"
            df[col] = df[col].replace(0, np.nan)
            logger.debug(f"Replaced 0s with NaN in column '{col}'.")

        # 9. Handle missing values with imputation
        distance_columns = [f"distance_to_{area}" for area in MODEL_AREAS]
        imputer = SimpleImputer(strategy='median')
        df[distance_columns] = imputer.fit_transform(df[distance_columns])
        logger.debug("Imputed missing values in distance columns with median.")

        # 10. Define feature columns
        feature_columns = distance_columns + ['hour', 'day_of_week', 'device_id']
        
        # 11. Check for any remaining missing values in feature columns
        missing_features = df[feature_columns].isnull().any()
        if missing_features.any():
            missing_cols = missing_features[missing_features].index.tolist()
            logger.error(f"Missing values detected in feature columns: {missing_cols}")
            # Fill remaining missing values with median or another strategy
            imputer = SimpleImputer(strategy='median')
            df[missing_cols] = imputer.fit_transform(df[missing_cols])
            logger.info(f"Filled missing values in feature columns: {missing_cols}")

        # 12. Define X and y
        X = df[feature_columns]
        y = df['area_id']

        # 13. Filter out invalid 'area_id' entries
        valid_indices = y != -1
        valid_sample_count = valid_indices.sum()
        if valid_sample_count == 0:
            logger.error("No valid samples found after filtering 'area_id'.")
            return None, None, None
        X = X[valid_indices]
        y = y[valid_indices]
        logger.info(f"Preprocessed data: {len(X)} samples ready for training.")
        return X, y, area_categories
    except Exception as e:
        logger.error(f"Error in preprocess_data: {e}")
        logger.debug(traceback.format_exc())
        # It's important to return None or appropriate values in case of failure
        return None, None, None

# Function to train the model
def train_model(X, y, area_categories):
    """
    Trains the XGBoost classifier.
    Returns the trained model.
    """
    logger = logging.getLogger('ai_agent')
    try:
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Initialize the XGBoost classifier
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )

        # Train the model
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=area_categories)
        logger.info(f"Classification Report:\n{report}")

        # Save the model
        joblib.dump(model, model_file)
        logger.info(f"Model trained and saved to {model_file}")

        return model
    except Exception as e:
        logger.error(f"Failed to train model: {e}")
        logger.debug(traceback.format_exc())
        return None

# Run the main coroutine
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("AI agent stopped by user.")
    except Exception as e:
        print(f"Unhandled exception: {e}")