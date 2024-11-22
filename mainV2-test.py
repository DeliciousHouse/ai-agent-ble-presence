import unittest
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import patch, AsyncMock, Mock
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Import the module to be tested
from mainV2 import (
    initialize_csv, get_local_time, fetch_user_name, log_sensor_data, call_service,
    log_automation_action, log_manual_light_event, log_location,
    get_dynamic_brightness_and_temperature, get_area_actions,
    extract_device_and_area_from_entity_id, prepare_features, determine_floor_and_section,
    is_area_overridden, handle_area_change, log_override_event, set_override,
    reset_override, cleanup_old_data, analyze_data, DEVICE_NAMES, MODEL_AREAS, HEADERS,
    automation_event_log_file, manual_event_log_file, sensor_data_log_file,
    override_log_file, analysis_output_file, model_file, sensor_data, sensor_data_lock,
    override_timers, monitor_overrides
)


# Mocking environment variables and dependencies
class MockedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime(2024, 1, 27, 10, 0, 0, tzinfo=tz)  # Example: 10 AM


@patch('mainV2.datetime', MockedDatetime)
class TestMainV2(unittest.IsolatedAsyncioTestCase):  # Use IsolatedAsyncioTestCase

    def setUp(self):
        # Clear shared data before each test
        sensor_data.clear()
        override_timers.clear()
        # Initialize logger mock
        self.logger = Mock()

    def test_initialize_csv_new_file(self):
        with patch('mainV2.os.path.isfile', return_value=False):
            with patch('mainV2.pd.DataFrame.to_csv') as mock_to_csv:
                initialize_csv(self.logger)
                mock_to_csv.assert_called_once()
                self.logger.info.assert_called()

    def test_initialize_csv_existing_file(self):
        with patch('mainV2.os.path.isfile', return_value=True):
            with patch('mainV2.pd.read_csv', return_value=pd.DataFrame()):
                with patch('mainV2.pd.DataFrame.to_csv') as mock_to_csv:
                    initialize_csv(self.logger)
                    mock_to_csv.assert_called_once()

    def test_initialize_csv_add_missing_columns(self):
        existing_data = {'timestamp': [1], 'device': [2]}
        existing_df = pd.DataFrame(existing_data)
        with patch('mainV2.os.path.isfile', return_value=True):
            with patch('mainV2.pd.read_csv', return_value=existing_df):
                with patch('mainV2.pd.DataFrame.to_csv') as mock_to_csv:
                    initialize_csv(self.logger)
                    mock_to_csv.assert_called_once()
                    self.logger.info.assert_called()


    def test_get_local_time(self):
        expected_time = datetime(2024, 1, 27, 10, 0, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
        self.assertEqual(get_local_time(), expected_time)

    @patch('mainV2.requests.get')
    def test_fetch_user_name(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"name": "Test User"}
        mock_get.return_value = mock_response
        user_name = fetch_user_name(self.logger, "user_id")
        self.assertEqual(user_name, "Test User")

    @patch('mainV2.pd.DataFrame.to_csv')
    def test_log_sensor_data(self, mock_to_csv):
        log_sensor_data(self.logger, "test_device", "test_area", {"distance_to_test_area": 1.0})
        mock_to_csv.assert_called_once()
        self.logger.debug.assert_called()

    @patch('mainV2.requests.post')
    async def test_call_service_success(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        result = await call_service(self.logger, "light", "turn_on", {"entity_id": "light.test"})
        self.assertTrue(result)
        self.logger.info.assert_called()

    @patch('mainV2.requests.post')
    async def test_call_service_failure(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Error"
        mock_post.return_value = mock_response

        # Patch HOME_ASSISTANT_CLOUD_API_URL to simulate a secondary API URL
        with patch.dict('mainV2.os.environ', {'HOME_ASSISTANT_CLOUD_API_URL': 'http://secondary_api'}):
            result = await call_service(self.logger, "light", "turn_on", {"entity_id": "light.test"})
        self.assertFalse(result)
        self.logger.error.assert_called()  # Expect multiple error logs due to both URLs failing

    @patch('mainV2.pd.DataFrame.to_csv')
    async def test_log_automation_action(self, mock_to_csv):
        await log_automation_action(self.logger, "light.test", "turned_on", brightness=100)
        mock_to_csv.assert_called_once()
        self.logger.info.assert_called()

    @patch('mainV2.pd.DataFrame.to_csv')
    async def test_log_manual_light_event(self, mock_to_csv):
        await log_manual_light_event(self.logger, "light.test", "turned_off", user="test_user")
        mock_to_csv.assert_called_once()
        self.logger.info.assert_called()



    @patch('mainV2.aiofiles.open', new_callable=AsyncMock)
    async def test_log_location(self, mock_aiofiles_open):
        mock_file = AsyncMock()
        mock_aiofiles_open.return_value = mock_file
        await log_location(self.logger, "test_device", "test_area", "enter")
        mock_file.__aenter__.assert_awaited_once()
        mock_file.__aexit__.assert_awaited_once()
        mock_file.write.assert_awaited_once()


    def test_get_dynamic_brightness_and_temperature(self):
        # Test at 10 AM (morning)
        with patch('mainV2.datetime', MockedDatetime):  # Use the mocked time
            brightness, color_temp = get_dynamic_brightness_and_temperature()
            self.assertEqual(brightness, 200)
            self.assertEqual(color_temp, 4000)

        # Test other times similarly
        with patch('mainV2.get_local_time', return_value=datetime(2024, 1, 27, 15, 0, 0)):  # 3 PM
            brightness, color_temp = get_dynamic_brightness_and_temperature()
            self.assertEqual(brightness, 255)
            self.assertEqual(color_temp, 3500)


    def test_get_area_actions_daytime(self):
        with patch('mainV2.get_local_time', return_value=datetime(2024, 1, 27, 10, 0, 0)):  # 10 AM
            area_actions = get_area_actions()
            for area in MODEL_AREAS:
                if area not in ["master_bedroom", "master_bathroom", "office", "sky_floor", "garage"]:
                    self.assertEqual(area_actions[area]["enter"][0][1], "turn_off")
                    self.assertEqual(area_actions[area]["exit"], [])
                else:
                    self.assertEqual(area_actions[area]["enter"][0][1], "turn_on")
                    self.assertEqual(area_actions[area]["exit"][0][1], "turn_off")

    def test_get_area_actions_nighttime(self):
        with patch('mainV2.get_local_time', return_value=datetime(2024, 1, 27, 23, 0, 0)):  # 11 PM
            area_actions = get_area_actions()
            for area in MODEL_AREAS:
                self.assertEqual(area_actions[area]["enter"][0][1], "turn_on")
                self.assertEqual(area_actions[area]["exit"][0][1], "turn_off")


    def test_extract_device_and_area_from_entity_id(self):
        entity_id = "sensor.bkam_iphone_ble_distance_to_lounge"
        device, area = extract_device_and_area_from_entity_id(entity_id)
        self.assertEqual(device, "bkam_iphone")
        self.assertEqual(area, "lounge")

        entity_id = "sensor.invalid_entity"
        device, area = extract_device_and_area_from_entity_id(entity_id)
        self.assertIsNone(device)
        self.assertIsNone(area)

    def test_prepare_features(self):
        sensor_data.update({
            "sensor.bkam_iphone_ble_distance_to_lounge": 1.0,
            "sensor.bkam_iphone_ble_distance_to_kitchen": 5.0
        })
        features = prepare_features(sensor_data, "bkam_iphone", MODEL_AREAS)
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape, (1, 10))  # 7 areas + hour + day_of_week + device_id



    def test_determine_floor_and_section(self):
        floor, section = determine_floor_and_section("lounge")
        self.assertEqual(floor, "first_floor")
        self.assertEqual(section, "middle")


        floor, section = determine_floor_and_section("unknown_area")
        self.assertIsNone(floor)
        self.assertIsNone(section)

    def test_is_area_overridden(self):
        override_timers["test_area"] = {"end_time": time.time() + 60}
        self.assertTrue(is_area_overridden("test_area"))
        override_timers["test_area"] = {"end_time": time.time() - 60}
        self.assertFalse(is_area_overridden("test_area"))

    @patch('mainV2.call_service', new_callable=AsyncMock)
    @patch('mainV2.log_location', new_callable=AsyncMock)
    async def test_handle_area_change(self, mock_log_location, mock_call_service):
        AREA_ACTIONS = get_area_actions()
        await handle_area_change(self.logger, "bkam_iphone", "lounge", "enter", AREA_ACTIONS)
        mock_call_service.assert_awaited()  # Check if call_service is called
        mock_log_location.assert_awaited()

    @patch('mainV2.pd.DataFrame.to_csv')
    async def test_log_override_event(self, mock_to_csv):
        await log_override_event(self.logger, "test_area", "light.test_area_lights", user="test_user")
        mock_to_csv.assert_called_once()

    @patch('mainV2.asyncio.get_event_loop')
    def test_set_override(self, mock_get_event_loop):
        mock_loop = Mock()
        mock_get_event_loop.return_value = mock_loop
        set_override("test_area", self.logger)
        self.assertIn("test_area", override_timers)
        mock_loop.create_task.assert_called_once()

    @patch('mainV2.pd.read_csv')
    @patch('mainV2.pd.DataFrame.to_csv')
    async def test_cleanup_old_data(self, mock_to_csv, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({'timestamp': [get_local_time()]})
        await cleanup_old_data(self.logger)
        mock_to_csv.assert_called()
        self.logger.info.assert_called()

    # Requires further refinement with proper mocking or a smaller, focused analysis function
    @patch('mainV2.cleanup_old_data', new_callable=AsyncMock)
    @patch('mainV2.os.path.isfile', return_value=True)  # Mock file existence
    @patch('mainV2.pd.read_csv', return_value=pd.DataFrame({'timestamp': [], 'device': [], 'estimated_area': []}))  # Mock DataFrame
    @patch('mainV2.joblib.dump') # Mock model saving
    @patch('mainV2.aiofiles.open', new_callable=AsyncMock)  # Mock writing report
    async def test_analyze_data_no_data(self, mock_open, mock_joblib, mock_pd, mock_isfile, mock_cleanup):
        await analyze_data(self.logger)
        self.logger.info.assert_called_with("No data available for analysis")  # No valid data


import unittest
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import patch, AsyncMock, Mock, MagicMock
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import json

# ... other imports

@patch('mainV2.datetime', MockedDatetime)
class TestMainV2(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # ... (Other setup remains the same)
        self.tasks = [] # To store all the tasks

    async def asyncTearDown(self): # Use Async Teardown!
        # Loop and cancel each task:
        for task in self.tasks:
            if task:
                task.cancel() # signal cancellation first
                try:
                     await task
                except asyncio.CancelledError:
                    pass  # Suppress expected cancelled errors from infinite loops

        await super().asyncTearDown() # Important


    @patch('mainV2.websockets.connect', new_callable=AsyncMock)
    async def test_monitor_overrides_authentication_success(self, mock_websocket_connect):
        mock_websocket = AsyncMock()
        mock_websocket_connect.return_value = mock_websocket
        mock_websocket.recv.side_effect = [
            json.dumps({"type": "auth_ok"}).encode(),
        ]


        with patch('mainV2.fetch_user_name', return_value="Test User"):
            task = asyncio.create_task(monitor_overrides(self.logger))
            self.tasks.append(task) # Keep track of the tasks for clean cancellation!
            await asyncio.sleep(0.1)  # Give it tiny time to start (adjust if needed)



# Example: Apply the same task cancellation approach to your other tests 
# (test_monitor_light_events, test_sensor_data_updater, etc.).
