import asyncio
import websockets
import json

async def test_light_control():
    uri = "ws://192.168.86.91:8123/api/websocket"
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI2ZjM0YzlmMDAxNjU0NjQ5OTUwZWQxZmMzN2Q3OTkzMSIsImlhdCI6MTczMDg1OTg4NCwiZXhwIjoyMDQ2MjE5ODg0fQ.RBnB6E27oWzRyYiDQC8IfMZMkl3GdUE4I96QgZYtjWk"

    async with websockets.connect(uri) as websocket:
        # Authenticate
        await websocket.send(json.dumps({
            "type": "auth",
            "access_token": token
        }))
        response = await websocket.recv()
        print("Auth response:", response)

        # Send turn_off command
        await websocket.send(json.dumps({
            "id": 6,
            "type": "call_service",
            "domain": "light",
            "service": "turn_off",
            "service_data": {
                "entity_id": "light.led_lights"
            }
        }))
        response = await websocket.recv()
        print("Service call response:", response)

asyncio.run(test_light_control())