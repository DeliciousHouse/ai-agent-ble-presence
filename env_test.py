from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Test environment variables
HOME_ASSISTANT_WS_URL = os.getenv("HOME_ASSISTANT_WS_URL")
LONG_LIVED_ACCESS_TOKEN = os.getenv("HA_LONG_LIVED_ACCESS_TOKEN")

if not LONG_LIVED_ACCESS_TOKEN:
    raise ValueError("Home Assistant Long-Lived Access Token not set.")

print(f"WebSocket URL: {HOME_ASSISTANT_WS_URL}")
print(f"Access Token: {LONG_LIVED_ACCESS_TOKEN[:5]}...")  # Print only part for security