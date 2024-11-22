import pandas as pd
from datetime import datetime
import os


automation_event_log_file = "test_automation_events.csv"

def log_automation_action(entity_id, action):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "entity_id": entity_id,
        "action": action,
    }
    df = pd.DataFrame([log_entry])
    if not os.path.isfile(automation_event_log_file):
        df.to_csv(automation_event_log_file, index=False)
    else:
        df.to_csv(automation_event_log_file, mode='a', header=False, index=False)
    print(f"Logged automation action: {log_entry}")

log_automation_action("light.lounge_lights", "turned_off")