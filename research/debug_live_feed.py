
import sqlite3
import json
import os

DB_PATH = "lab_notebook.db"

def check_live_feed():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    print("Fetching last 50 live feed entries...")
    cursor.execute("""
        SELECT experiment_id, timestamp, metadata_json 
        FROM entries 
        WHERE entry_type='live_feed' AND experiment_id='86bc13d1-eb9'
        ORDER BY timestamp DESC 
    """)
    rows = cursor.fetchall()
    
    print(f"{'Experiment ID':<36} | {'Timestamp':<20} | {'Gen (extracted)'}")
    print("-" * 80)
    
    for row in rows:
        exp_id = row['experiment_id']
        ts = row['timestamp']
        meta_json = row['metadata_json']
        gen = "N/A"
        try:
            meta = json.loads(meta_json)
            payload = meta.get('payload', {})
            gen = payload.get('generation', "N/A")
            live_type = meta.get('live_feed_type', "N/A")
        except:
            gen = "Error parsing JSON"
            live_type = "Error"
            
        print(f"{exp_id:<36} | {ts:<20} | {live_type:<20} | {gen}")

    conn.close()

if __name__ == "__main__":
    check_live_feed()
