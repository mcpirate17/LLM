import time
import os
import hashlib

FILE_PATH = "REAL_TOKEN_EVAL_BRAINSTORM.md"

def get_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

if __name__ == "__main__":
    if not os.path.exists(FILE_PATH):
        print(f"File {FILE_PATH} not found.")
        exit(1)
    
    last_hash = get_hash(FILE_PATH)
    print(f"Monitoring {FILE_PATH} every 25 seconds...")
    
    try:
        while True:
            time.sleep(25)
            current_hash = get_hash(FILE_PATH)
            if current_hash != last_hash:
                print("CHANGE DETECTED")
                break
    except KeyboardInterrupt:
        print("Monitoring stopped.")
