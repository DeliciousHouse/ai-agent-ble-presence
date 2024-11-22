import requests
import os
from dotenv import load_dotenv

load_dotenv()

HOME_ASSISTANT_API_URL = os.getenv("HOME_ASSISTANT_API_URL")
HEADERS = {
    "Authorization": f"Bearer {os.getenv('HA_LONG_LIVED_ACCESS_TOKEN')}",
    "content-type": "application/json",
}

def test_call_service():
    service_url = f"{HOME_ASSISTANT_API_URL}/services/light/turn_on"
    service_data = {"entity_id": "light.test_light", "brightness": 200}
    try:
        response = requests.post(service_url, headers=HEADERS, json=service_data)
        print("Response:", response.status_code, response.text)
    except Exception as e:
        print(f"Service call failed: {e}")
test_call_service()