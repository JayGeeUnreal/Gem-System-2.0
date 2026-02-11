import requests
import time
import datetime
import random
import sys

# URL of your async server
SERVER_URL = "http://127.0.0.1:5000/chat"

print(f"--- Auto Tester (No Typing Needed) ---")
print(f"Target: {SERVER_URL}")
print("Press Ctrl+C to stop.\n")

counter = 1
messages = ["Hello", "Test", "Stream data", "Another msg", "Async is cool"]

while True:
    try:
        # Create data
        msg = random.choice(messages)
        payload = {
            "user": f"User_{random.randint(1,99)}",
            "message": f"{msg} ({counter})",
            "platform": "conda_test",
            "timestamp": str(datetime.datetime.now())
        }

        # Send
        print(f"Sending message #{counter}...", end=" ", flush=True)
        try:
            resp = requests.post(SERVER_URL, json=payload, timeout=2)
            if resp.status_code == 200:
                print("SUCCESS")
            else:
                print(f"FAIL ({resp.status_code})")
        except requests.exceptions.ConnectionError:
             print("FAIL (Connection Refused - Is server running?)")

        counter += 1
        time.sleep(2)

    except KeyboardInterrupt:
        print("\nStopping.")
        sys.exit(0)