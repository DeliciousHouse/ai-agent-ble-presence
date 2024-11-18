import asyncio
import json
import websockets
import requests
import time
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier  # Ensure XGBClassifier is imported
from src.data_preprocessing import load_data, preprocess_data
from src.model_training import train_model
from src.inference import predict_room
import traceback
from aiologger import Logger
from aiologger.handlers.files import AsyncFileHandler
from aiologger.handlers.streams import AsyncStreamHandler
from aiohttp import web
import aiofiles

# Load environment variables from .env file
load_dotenv()

# Configuration from environment variables
HOME_ASSISTANT_WS_URL = os.getenv("HOME_ASSISTANT_WS_URL")
HOME_ASSISTANT_WSS_URL = os.getenv("HOME_ASSISTANT_WSS_URL")
HOME_ASSISTANT_API_URL = os.getenv("HOME_ASSISTANT_API_URL")
HOME_ASSISTANT_CLOUD_API_URL = os.getenv("HOME_ASSISTANT_CLOUD_API_URL")
LONG_LIVED_ACCESS_TOKEN = os.getenv("HA_LONG_LIVED_ACCESS_TOKEN")

if not LONG_LIVED_ACCESS_TOKEN:
    raise ValueError("Home Assistant Long-Lived Access Token not set.")

# Headers for REST API calls
HEADERS = {
    "Authorization": f"Bearer {LONG_LIVED_ACCESS_TOKEN}",
    "content-type": "application/json",
}

# Define the device names and areas
DEVICE_NAMES = ["madisons_iphone", "bkam_iphone", "nova_iphone", "bkam_apple_watch"]
AREAS = [
    "lounge", "master_bedroom", "kitchen", "balcony", "garage",
    "office", "front_porch", "driveway",
    "nova_s_room", "christian_s_room", "sky_floor", "master_bathroom",
    "backyard", "dining_room",
    "laundry_room", "dressing_room", "bathroom"
]

BLE_DEVICES = [
    "master_bedroom",
    "master_bathroom",
    "lounge",
    "kitchen",
    "office",
    "sky_floor",
    "balcony"
]

# Log file paths
automation_event_log_file = "/app/data/automation_light_events.csv"
manual_event_log_file = "/app/data/manual_light_events.csv"
data_log_file = "/app/data/sensor_data.csv"
override_log_file = "/app/data/override_log.csv"
ai_agent_log_file = "/app/data/ai_agent.log"

# Initialize CSV with headers
async def initialize_csv(logger):
    headers = ["timestamp", "device", "estimated_room"] + [
        f"distance_to_{area}" for area in AREAS
    ]
    try:
        if not os.path.isfile(data_log_file):
            pd.DataFrame(columns=headers).to_csv(data_log_file, index=False)
            await logger.info(f"Initialized CSV with headers: {headers}")
    except Exception as e:
        await logger.error(f"Failed to initialize CSV: {e}")

# Log sensor data to the CSV
async def log_sensor_data(logger, device, estimated_room, distances):
    timestamp = datetime.now().isoformat()
    data_entry = {"timestamp": timestamp, "device": device, "estimated_room": estimated_room}
    for area, distance in distances.items():
        data_entry[f"distance_to_{area}"] = distance
    try:
        df = pd.DataFrame([data_entry])
        if not os.path.isfile(data_log_file):
            df.to_csv(data_log_file, index=False)
        else:
            df.to_csv(data_log_file, mode='a', header=False, index=False)
        await logger.debug(f"Logged sensor data to {data_log_file}: {data_entry}")
    except Exception as e:
        await logger.error(f"Failed to log sensor data: {e}")

# Global state
current_rooms = {}
last_action_time = {}
override_timers = {}  # Stores override timers for rooms

# Data analysis settings
data_analysis_interval = 900  # Analyze data every 15 minutes
min_time_between_actions = 10  # Minimum time between actions per device (seconds)
override_duration = 600  # Override duration in seconds (e.g., 10 minutes)
MAX_DISTANCE = 20  # Adjust based on maximum expected distance
MIN_REQUIRED_SENSORS = 2  # Minimum number of sensors required to make a prediction

# Helper function to determine brightness and color temperature based on time and ambient light
def get_dynamic_brightness_and_temperature(sensor_data):
    current_hour = datetime.now().hour
    try:
        ambient_light = float(sensor_data.get("sensor.curtain_light_level", 0))
    except (ValueError, TypeError):
        ambient_light = 0  # Default if sensor data isn't available or invalid

    # Define brightness and temperature based on time and ambient light
    if 6 <= current_hour < 12:  # Morning
        brightness = int(max(100, min(255, 255 * (1 - ambient_light / 100))))
        color_temp = 4000  # Cool white for morning
    elif 12 <= current_hour < 18:  # Afternoon
        brightness = int(max(150, min(255, 255 * (1 - ambient_light / 100))))
        color_temp = 3500  # Natural white for afternoon
    elif 18 <= current_hour < 22:  # Evening
        brightness = int(max(50, min(180, 180 * (1 - ambient_light / 100))))
        color_temp = 3000  # Warm white for evening
    else:  # Nighttime
        brightness = int(max(20, min(80, 80 * (1 - ambient_light / 100))))
        color_temp = 2700  # Very warm white for nighttime

    return brightness, color_temp

# Map areas to BLE sensors
AREA_BLE_SENSORS = {area: f"sensor.{area}_ble" for area in AREAS}

# Function to dynamically generate AREA_ACTIONS based on sensor_data
def get_area_actions(sensor_data):
    brightness, color_temp = get_dynamic_brightness_and_temperature(sensor_data)
    area_actions = {
        area: {
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
        } for area in AREAS
    }
    return area_actions

# Function to fetch live sensor data via websocket
async def get_live_sensor_data(logger):
    sensor_data = {}  # Initialize sensor_data
    for ws_url in [HOME_ASSISTANT_WS_URL, HOME_ASSISTANT_WSS_URL]:
        try:
            async with websockets.connect(ws_url, timeout=10) as websocket:
                # Authentication
                await websocket.send(json.dumps({"type": "auth", "access_token": LONG_LIVED_ACCESS_TOKEN}))
                # Subscribe to state_changed events
                await websocket.send(json.dumps({"id": 5, "type": "subscribe_events", "event_type": "state_changed"}))  # Unique id
                while True:
                    message = await websocket.recv()
                    try:
                        event = json.loads(message)
                        if event.get("type") == "event" and event["event"].get("event_type") == "state_changed":
                            entity_id = event["event"]["data"]["entity_id"]
                            if entity_id == "sensor.curtain_light_level":
                                new_state = event["event"]["data"]["new_state"]["state"]
                                sensor_data[entity_id] = new_state
                                await logger.debug(f"Updated sensor_data: {sensor_data}")
                                return sensor_data  # Return the updated data
                    except json.JSONDecodeError as e:
                        await logger.error(f"JSON decode error: {e}")
                        continue
        except (websockets.exceptions.ConnectionClosedError, websockets.exceptions.InvalidStatusCode) as e:
            await logger.error(f"Connection error with {ws_url} in get_live_sensor_data: {e}")
            await asyncio.sleep(10)
            continue
        except Exception as e:  # Catch other websocket errors
            await logger.error(f"Error getting live sensor data with {ws_url}: {e}")
            await asyncio.sleep(10)
            continue

# Async function to call Home Assistant services with dynamic brightness and color_temp
async def call_service(logger, domain, service, service_data):
    for api_url in [HOME_ASSISTANT_API_URL, HOME_ASSISTANT_CLOUD_API_URL]:
        url = f"{api_url}/services/{domain}/{service}"
        try:
            response = requests.post(url, headers=HEADERS, json=service_data, timeout=5)
            if response.status_code == 200:
                await logger.info(f"Called service {domain}.{service} with {service_data} using {api_url}")
                return
            else:
                await logger.error(f"Failed to call service {domain}.{service} using {api_url}: {response.text}")
        except requests.exceptions.RequestException as e:
            await logger.error(f"Error calling service {domain}.{service} using {api_url}: {e}")
    await logger.error(f"All attempts to call service {domain}.{service} failed.")

# Async function to log automation actions with latest sensor data
async def log_automation_action(logger, entity_id, action, brightness=None):
    current_time = datetime.now()
    hour = current_time.hour
    # Fetch latest sensor data
    sensor_data = await get_live_sensor_data(logger)
    try:
        ambient_light = float(sensor_data.get("sensor.curtain_light_level", 0))
    except ValueError:
        ambient_light = 0  # Default if sensor data isn't available

    log_entry = {
        "timestamp": current_time.isoformat(),
        "entity_id": entity_id,
        "action": action,
        "brightness": brightness,
        "hour": hour,
        "ambient_light": ambient_light,
    }

    # Append log entry to the CSV file
    try:
        df = pd.DataFrame([log_entry])
        if not os.path.isfile(automation_event_log_file):
            df.to_csv(automation_event_log_file, index=False)
        else:
            df.to_csv(automation_event_log_file, mode='a', header=False, index=False)
        await logger.info(f"Logged automation action: {log_entry}")
    except Exception as e:
        await logger.error(f"Failed to log automation action: {e}")

# Async function to monitor automation events
async def monitor_automation_events(logger):
    for ws_url in [HOME_ASSISTANT_WS_URL, HOME_ASSISTANT_WSS_URL]:
        try:
            async with websockets.connect(ws_url, timeout=10) as websocket:
                await websocket.send(json.dumps({"type": "auth", "access_token": LONG_LIVED_ACCESS_TOKEN}))
                await websocket.send(json.dumps({"id": 100, "type": "subscribe_events", "event_type": "state_changed"}))  # Unique id
                while True:
                    message = await websocket.recv()
                    event = json.loads(message)
                    if event.get("type") == "event" and event["event"].get("event_type") == "state_changed":
                        entity_id = event["event"]["data"]["entity_id"]
                        automation_state = event["event"]["data"]["new_state"]["state"]
                        if entity_id.startswith("automation.") and automation_state == "on":
                            affected_light = entity_id.replace("automation.", "light.")
                            asyncio.create_task(log_automation_action(logger, affected_light, "triggered"))  # Asynchronous logging
        except Exception as e:
            await logger.error(f"Error monitoring automation events: {e}")
            await asyncio.sleep(10)  # Retry connection

# Async function to ensure critical area lighting based on presence and ambient light
async def ensure_critical_area_lighting(logger):
    while True:
        current_hour = datetime.now().hour
        is_dark = 18 <= current_hour or current_hour < 6  # Example darkness condition
        # Fetch latest sensor data
        sensor_data = await get_live_sensor_data(logger)
        try:
            ambient_light = float(sensor_data.get("sensor.curtain_light_level", 0))
        except (ValueError, TypeError):
            ambient_light = 0  # Default if sensor data isn't available or invalid

        for area, presence_sensor in [("balcony", "sensor.balcony_presence"), ("backyard", "sensor.backyard_presence")]:
            presence_state = sensor_data.get(presence_sensor, "off")
            light_entity = f"light.{area}_lights"

            if is_dark and presence_state == "on" and ambient_light < 30:  # Adjust ambient threshold as needed
                # Fetch latest brightness and color_temp
                brightness, color_temp = get_dynamic_brightness_and_temperature(sensor_data)
                service_data = {
                    "entity_id": light_entity,
                    "brightness": brightness,
                    "color_temp": color_temp
                }
                await call_service(logger, "light", "turn_on", service_data)
                await logger.info(f"Ensured {light_entity} is on due to presence and low light")

        await asyncio.sleep(60)  # Check every 60 seconds

# Async function to log manual light events with latest sensor data
async def log_manual_light_event(logger, entity_id, action, brightness=None):
    current_time = datetime.now()
    hour = current_time.hour
    # Fetch latest sensor data
    sensor_data = await get_live_sensor_data(logger)
    try:
        ambient_light = float(sensor_data.get("sensor.curtain_light_level", 0))
    except ValueError:
        ambient_light = 0  # Default if sensor data isn't available

    log_entry = {
        "timestamp": current_time.isoformat(),
        "entity_id": entity_id,
        "action": action,
        "brightness": brightness,
        "hour": hour,
        "ambient_light": ambient_light,
    }

    try:
        df = pd.DataFrame([log_entry])
        if not os.path.isfile(manual_event_log_file):
            df.to_csv(manual_event_log_file, index=False)
        else:
            df.to_csv(manual_event_log_file, mode='a', header=False, index=False)
        await logger.info(f"Logged manual light event: {log_entry}")
    except Exception as e:
        await logger.error(f"Failed to log manual light event: {e}")

# Async function to monitor manual light events
async def monitor_light_events(logger):
    for ws_url in [HOME_ASSISTANT_WS_URL, HOME_ASSISTANT_WSS_URL]:
        try:
            async with websockets.connect(ws_url, timeout=10) as websocket:
                await websocket.send(json.dumps({"type": "auth", "access_token": LONG_LIVED_ACCESS_TOKEN}))
                await websocket.send(json.dumps({"id": 101, "type": "subscribe_events", "event_type": "state_changed"}))  # Unique id
                while True:
                    message = await websocket.recv()
                    event = json.loads(message)
                    if event.get("type") == "event" and event["event"].get("event_type") == "state_changed":
                        entity_id = event["event"]["data"]["entity_id"]
                        new_state = event["event"]["data"]["new_state"]
                        old_state = event["event"]["data"]["old_state"]

                        if entity_id.startswith("light."):
                            if new_state["state"] != old_state["state"]:
                                action = "turned_on" if new_state["state"] == "on" else "turned_off"
                                asyncio.create_task(log_manual_light_event(logger, entity_id, action))  # Asynchronous logging
                            elif new_state["attributes"].get("brightness") != old_state["attributes"].get("brightness"):
                                action = "brightness_changed"
                                brightness = new_state["attributes"].get("brightness")
                                asyncio.create_task(log_manual_light_event(logger, entity_id, action, brightness))  # Asynchronous logging
        except Exception as e:
            await logger.error(f"Error monitoring light events with {ws_url}: {e}")
            await asyncio.sleep(10)  # Retry connection

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

def determine_floor_and_section(room):
    for floor, sections in FLOOR_MAP.items():
        for section, rooms in sections.items():
            if room in rooms:
                return floor, section
    return None, None

# Async function to log location
async def log_location(logger, device, room, action):
    timestamp = datetime.now().isoformat()
    log_entry = {"timestamp": timestamp, "device": device, "room": room, "action": action}
    log_dir = "/app/data"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "location_log.json")
    try:
        async with aiofiles.open(log_file, "a") as logfile:
            await logfile.write(json.dumps(log_entry) + "\n")
        await logger.debug(f"Logged location: {log_entry}")
    except Exception as e:
        await logger.error(f"Failed to write log entry: {e}")

# Function to check if a room is currently overridden
def is_room_overridden(room):
    # Check override_timers dictionary
    current_time = time.time()
    override_info = override_timers.get(room)
    return override_info is not None and override_info['end_time'] > current_time

# Async function to handle room changes
async def handle_room_change(logger, device, room, action_type, AREA_ACTIONS):
    current_time = time.time()
    if last_action_time.get(device, 0) + min_time_between_actions > current_time:
        await logger.debug(f"Skipping action for {device} {action_type}ing {room} due to debounce.")
        return
    # Fetch latest sensor data
    sensor_data = await get_live_sensor_data(logger)
    brightness, color_temp = get_dynamic_brightness_and_temperature(sensor_data)

    await logger.info(f"{device} {action_type}ing room: {room}")
    actions = AREA_ACTIONS.get(room, {}).get(action_type, [])
    current_floor, current_section = determine_floor_and_section(room)
    prev_room = current_rooms.get(device)
    if prev_room:
        prev_floor, prev_section = determine_floor_and_section(prev_room)
        if current_floor != prev_floor:
            await logger.warning(f"Invalid transition detected: {prev_room} (floor {prev_floor}) to {room} (floor {current_floor}). Skipping.")
            return 
        elif current_section != prev_section:
            await logger.info(f"Transition within the same floor: {prev_room} to {room}. Proceeding.") 

    if is_room_overridden(room):
        await logger.info(f"Room {room} is currently overridden. Skipping {action_type} actions.")
        return

    for domain, service, service_data in actions:
        # Dynamically include brightness and color_temp if action is 'turn_on'
        if service == "turn_on":
            service_data["brightness"] = brightness
            service_data["color_temp"] = color_temp
        await call_service(logger, domain, service, service_data)
    await log_location(logger, device, room, action_type)
    last_action_time[device] = current_time
    current_rooms[device] = room

# Async function to log override events
async def log_override_event(logger, room, light_entity_id, user=None):
    timestamp = datetime.now().isoformat()
    override_entry = {
        "timestamp": timestamp,
        "room": room,
        "light_entity_id": light_entity_id,
        "user": user or "unknown",
        "hour": datetime.now().hour,
        "day_of_week": datetime.now().weekday(),
    }
    try:
        df = pd.DataFrame([override_entry])
        if not os.path.isfile(override_log_file):
            df.to_csv(override_log_file, index=False)
        else:
            df.to_csv(override_log_file, mode='a', header=False, index=False)
        await logger.debug(f"Logged override event: {override_entry}")
    except Exception as e:
        await logger.error(f"Failed to log override event: {e}")

# Function to set override
def set_override(room, logger):
    override_end_time = time.time() + override_duration
    override_timers[room] = {'end_time': override_end_time}
    # Schedule the reset_override coroutine
    loop = asyncio.get_event_loop()
    loop.create_task(reset_override(logger, room))

# Async function to reset override
async def reset_override(logger, room):
    await asyncio.sleep(override_duration)
    override_timers.pop(room, None)
    await logger.info(f"Override for room {room} has been reset.")

# Async function to cleanup old data
async def cleanup_old_data(logger, retention_days=30):
    for file in [data_log_file, override_log_file]:
        if os.path.isfile(file):
            try:
                # Read CSV with error handling for bad lines
                df = pd.read_csv(file, parse_dates=['timestamp'], on_bad_lines='skip')
                cutoff_date = datetime.now() - timedelta(days=retention_days)
                df = df[df['timestamp'] >= cutoff_date]
                df.to_csv(file, index=False)
                await logger.info(f"Cleaned up data in {file} older than {retention_days} days.")
            except Exception as e:
                await logger.error(f"Failed to clean up old data in {file}: {e}")

# Async function to analyze data (previously in a separate thread)
async def analyze_data(logger):
    await logger.info("Starting advanced data analysis coroutine")
    analysis_output_file = "/app/data/analysis_output.txt"
    model_file = "/app/src/models/ble_presence_model.pkl"
    retention_days = 30
    while True:
        try:
            await cleanup_old_data(logger, retention_days=retention_days)
            if os.path.isfile(data_log_file):
                try:
                    # Read CSV with error handling for bad lines
                    df = pd.read_csv(data_log_file, parse_dates=['timestamp'], on_bad_lines='skip')
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    # Proceed with data analysis...
                    df.dropna(inplace=True)
                    df['hour'] = df['timestamp'].dt.hour
                    df['day_of_week'] = df['timestamp'].dt.dayofweek

                    # Encode categorical variables
                    df['device_id'] = df['device'].astype('category').cat.codes
                    device_categories = df['device'].astype('category').cat.categories.tolist()

                    # Encode rooms
                    df['room_id'] = df['estimated_room'].apply(lambda x: AREAS.index(x) if x in AREAS else -1)
                    room_categories_list = AREAS  # Avoid overwriting the global room_categories

                    # Feature columns
                    distance_columns = [col for col in df.columns if col.startswith('distance_to_')]
                    distance_columns.sort()

                    X = df[distance_columns + ['hour', 'day_of_week', 'device_id']]
                    y = df['room_id']

                    valid_indices = y != -1
                    X = X[valid_indices]
                    y = y[valid_indices]

                    if X.empty:
                        await logger.warning("No valid data available for training after filtering. Skipping model training.")
                    else:
                        feature_names = X.columns.tolist()

                        # Train the model
                        model = XGBClassifier(
                            n_estimators=100,
                            max_depth=6,
                            learning_rate=0.1,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            random_state=42
                        )
                        model.fit(X, y)

                        y_pred = model.predict(X)
                        report = classification_report(y, y_pred, target_names=room_categories_list)

                        # Save the model
                        joblib.dump(model, model_file)

                        # Write analysis report
                        async with aiofiles.open(analysis_output_file, 'w') as f:
                            await f.write("Classification Report:\n")
                            await f.write(report)
                        await logger.info(f"Advanced analysis results written to {analysis_output_file}")
                except Exception as e:
                    await logger.error(f"Error during data analysis: {e}")
            else:
                await logger.info("No data available for analysis")
        except Exception as e:
            await logger.error(f"Error during advanced data analysis: {e}")
            await logger.debug(traceback.format_exc())
        await asyncio.sleep(data_analysis_interval)

# Async function to monitor overrides
async def monitor_overrides(logger):
    for ws_url in [HOME_ASSISTANT_WS_URL, HOME_ASSISTANT_WSS_URL]:
        try:
            await logger.info(f"Attempting to connect to {ws_url} for override monitoring")
            async with websockets.connect(ws_url, timeout=10) as websocket:
                await logger.info(f"Connected to {ws_url} for override monitoring")

                # Authentication
                auth_required_message = await websocket.recv()
                auth_required = json.loads(auth_required_message)
                await logger.debug(f"Received auth required message: {auth_required}")

                if auth_required.get('type') != 'auth_required':
                    await logger.error(f"Expected 'auth_required' message but received: {auth_required}")
                    continue

                await logger.debug("Sending authentication request...")
                await websocket.send(
                    json.dumps({"type": "auth", "access_token": LONG_LIVED_ACCESS_TOKEN})
                )

                auth_response = await websocket.recv()
                auth_result = json.loads(auth_response)
                await logger.debug(f"Authentication response: {auth_result}")

                if auth_result.get("type") != "auth_ok":
                    await logger.error(f"Authentication failed with {ws_url}: {auth_result.get('message', '')}")
                    continue
                await logger.info(f"Authenticated successfully with Home Assistant using {ws_url}")

                # Subscribe to state_changed events
                await websocket.send(
                    json.dumps({"id": 200, "type": "subscribe_events", "event_type": "state_changed"})
                )  # Unique id
                await logger.debug("Subscribed to state_changed events for override monitoring")

                while True:
                    message = await websocket.recv()
                    event = json.loads(message)
                    if event.get("type") == "event" and event["event"].get("event_type") == "state_changed":
                        entity_id = event["event"]["data"]["entity_id"]
                        new_state = event["event"]["data"]["new_state"]["state"]
                        old_state = event["event"]["data"]["old_state"]["state"]

                        for area in AREAS:
                            light_entity_id = f"light.{area}_lights"  # Corrected entity_id format
                            if entity_id == light_entity_id:
                                if old_state == "on" and new_state == "off":
                                    await logger.info(f"Manual override detected for {light_entity_id}")
                                    await log_override_event(logger, room=area, light_entity_id=light_entity_id)
                                    set_override(area, logger)
                                elif old_state == "off" and new_state == "on":
                                    await logger.info(f"Manual override turned on for {light_entity_id}")
                                    await log_override_event(logger, room=area, light_entity_id=light_entity_id, user="manual_on")
                                    set_override(area, logger)
        except Exception as e:
            await logger.error(f"An error occurred during override monitoring with {ws_url}: {e}")
            await logger.debug(traceback.format_exc())
            await asyncio.sleep(10)  # Wait before retrying

# Async function to monitor room location
async def monitor_room_location(logger, model):
    # Schedule analyze_data as a background task
    asyncio.create_task(analyze_data(logger))

    # Start asynchronous tasks concurrently
    asyncio.create_task(monitor_overrides(logger))
    asyncio.create_task(ensure_critical_area_lighting(logger))
    asyncio.create_task(monitor_automation_events(logger))
    asyncio.create_task(monitor_light_events(logger))

    model_file = "/app/src/models/ble_presence_model.pkl"
    model_loaded = False
    if os.path.isfile(model_file):
        try:
            model = joblib.load(model_file)
            if not isinstance(model, XGBClassifier):
                raise TypeError("'XGBClassifier' object is not subscriptable")
            await logger.info("Loaded trained model for real-time predictions.")
            model_loaded = True
        except Exception as e:
            await logger.error(f"Error loading model: {e}")
            model = None
    else:
        model = None
        await logger.warning("Model file not found. Real-time predictions will not be available.")

    device_ble_distances = {device: {} for device in DEVICE_NAMES}

    while True:
        # Fetch the latest sensor data
        sensor_data = await get_live_sensor_data(logger)
        AREA_ACTIONS = get_area_actions(sensor_data)  # Update AREA_ACTIONS with current brightness/color_temp

        # Connect to websocket and process messages
        for ws_url in [HOME_ASSISTANT_WS_URL, HOME_ASSISTANT_WSS_URL]:
            try:
                async with websockets.connect(ws_url, timeout=10) as websocket:
                    await websocket.send(json.dumps({"type": "auth", "access_token": LONG_LIVED_ACCESS_TOKEN}))
                    await websocket.send(json.dumps({"id": 300, "type": "subscribe_events", "event_type": "state_changed"}))  # Unique id

                    while True:
                        message = await websocket.recv()
                        event = json.loads(message)
                        if event.get("type") == "event" and event["event"].get("event_type") == "state_changed":
                            entity_id = event["event"]["data"]["entity_id"]
                            new_state = event["event"]["data"]["new_state"]["state"]

                            # Update sensor_data based on entity_id
                            if entity_id == "sensor.curtain_light_level":
                                sensor_data["sensor.curtain_light_level"] = new_state
                            elif entity_id.startswith("sensor.") and entity_id.endswith("_distance"):
                                sensor_data[entity_id] = new_state
                            # Add more conditions if you have other sensor types

                            await logger.debug(f"Received sensor update: {entity_id} = {new_state}")

                            # Extract device from entity_id
                            device = extract_device_from_entity_id(entity_id)
                            if device is None:
                                await logger.warning(f"Could not extract device from entity_id: {entity_id}")
                                continue

                            # Prepare features for prediction
                            features = prepare_features(sensor_data, device, AREAS)

                            # Make prediction if model is available
                            if model_loaded and model:
                                try:
                                    predicted_room_idx = model.predict(features)[0]
                                    if predicted_room_idx == -1:
                                        await logger.warning(f"Invalid room prediction for device {device}.")
                                        continue
                                    estimated_room = AREAS[predicted_room_idx]

                                    if estimated_room:
                                        if current_rooms.get(device) != estimated_room:
                                            if current_rooms.get(device):
                                                # Handle exit from previous room
                                                asyncio.create_task(handle_room_change(logger, device, current_rooms[device], "exit", AREA_ACTIONS))  # Async handling
                                            # Handle entry to new room
                                            asyncio.create_task(handle_room_change(logger, device, estimated_room, "enter", AREA_ACTIONS))  # Async handling
                                except Exception as e:
                                    await logger.error(f"Error during room prediction for device {device}: {e}")
                            else:
                                await logger.debug("Model not loaded. Skipping room prediction.")
            except Exception as e:
                await logger.error(f"Error in monitor_room_location with {ws_url}: {e}")
                await asyncio.sleep(10)  # Retry connection

        await asyncio.sleep(1)  # Short delay to prevent tight loop

# Function to extract device name from entity_id
def extract_device_from_entity_id(entity_id):
    """
    Extracts the device name from the entity_id.
    Assumes entity_id contains the device name, e.g., 'sensor.madisons_iphone_distance'.
    """
    for device in DEVICE_NAMES:
        if device in entity_id:
            return device
    return None

# Function to prepare features for prediction
def prepare_features(sensor_data, device, areas):
    # Initialize a dictionary for features
    features = {}

    # Add distance features
    for area in areas:
        key = f"distance_to_{area}"
        # Assuming sensor_data contains distance metrics, adjust as needed
        try:
            features[key] = float(sensor_data.get(f"sensor.{area}_distance", 0))
        except ValueError:
            features[key] = 0  # Default if conversion fails

    # Add time-based features
    current_time = datetime.now()
    features['hour'] = current_time.hour
    features['day_of_week'] = current_time.weekday()

    # Add device ID as categorical
    device_id = device
    # Map device to an index (assuming device_categories are handled during model training)
    device_category_map = {device: idx for idx, device in enumerate(DEVICE_NAMES)}
    device_idx = device_category_map.get(device_id, -1)
    features['device_id'] = device_idx

    # Ensure all feature_names are present
    # Assuming feature order: ['hour', 'day_of_week'] + ['distance_to_<area>'] + ['device_id']
    feature_values = [features.get(name, 0) for name in ['hour', 'day_of_week'] + [f"distance_to_{area}" for area in areas] + ['device_id']]

    return np.array(feature_values).reshape(1, -1)  # Reshape for prediction

# Function to gracefully shutdown the logger
async def shutdown_logger(logger):
    await logger.shutdown()

# Health server implementation using aiohttp
async def health(request):
    return web.Response(text="OK")

async def start_health_server(logger):
    app = web.Application()
    app.router.add_get('/health', health)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8000)
    await site.start()
    await logger.info("Health server started on port 8000")

# Main application logic
async def main():
    # Initialize aiologger with asynchronous handlers
    logger = Logger.with_default_handlers(name='ai_agent')

    # Create and add file handler (asynchronous)
    file_handler = AsyncFileHandler(ai_agent_log_file)
    await logger.add_handler(file_handler)

    # Create and add stream handler (logs to STDOUT) (asynchronous)
    stream_handler = AsyncStreamHandler()
    await logger.add_handler(stream_handler)

    try:
        # Initialize CSV file
        await initialize_csv(logger)
        # Load and preprocess data
        df = await asyncio.to_thread(load_data)
        X, y, room_categories = await asyncio.to_thread(preprocess_data, df)
        # Train the model
        model = await asyncio.to_thread(train_model, X, y, room_categories)
        await logger.info("Model training completed.")

        # Start health server
        await start_health_server(logger)

        await logger.info("Starting AI agent...")
        await monitor_room_location(logger, model)  # Pass the model to the monitoring function
    except KeyboardInterrupt:
        await logger.info("AI agent stopped by user.")
    except Exception as e:
        await logger.error(f"Unhandled exception: {e}")
        await logger.debug(traceback.format_exc())
    finally:
        await shutdown_logger(logger)

# Run the main coroutine
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("AI agent stopped by user.")
    except Exception as e:
        print(f"Unhandled exception: {e}")