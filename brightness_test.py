from datetime import datetime

def get_dynamic_brightness_and_temperature(ambient_light):
    current_hour = datetime.now().hour
    if 6 <= current_hour < 12:  # Morning
        brightness = max(100, min(255, 255 * (1 - ambient_light / 100)))
        color_temp = 4000
    elif 12 <= current_hour < 18:  # Afternoon
        brightness = max(150, min(255, 255 * (1 - ambient_light / 100)))
        color_temp = 3500
    elif 18 <= current_hour < 22:  # Evening
        brightness = max(50, min(180, 180 * (1 - ambient_light / 100)))
        color_temp = 3000
    else:  # Nighttime
        brightness = max(20, min(80, 80 * (1 - ambient_light / 100)))
        color_temp = 2700
    return brightness, color_temp

# Test with ambient light levels
for ambient_light in [0, 30, 70, 100]:
    print(f"Ambient Light: {ambient_light} -> {get_dynamic_brightness_and_temperature(ambient_light)}")