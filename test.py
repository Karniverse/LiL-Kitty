import threading
import time

def worker():
    while True:
        print("working")
        time.sleep(1)

threading.Thread(target=worker, daemon=True).start()
print("🟢 System ready. Press Ctrl+C to exit.")

import time
while True:
    time.sleep(1)
