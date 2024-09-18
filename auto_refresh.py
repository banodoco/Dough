import os
import sys
import portalocker
import json
from flask import Flask, jsonify
import logging
import threading
import time
import random
from queue import Queue

from utils.constants import REFRESH_LOCK_FILE, REFRESH_PROCESS_PORT, REFRESH_TARGET_FILE

app = Flask(__name__)


target_file = REFRESH_TARGET_FILE
lock_file = REFRESH_LOCK_FILE
refresh_queue = Queue()
refresh_thread = None

last_refreshed_on = 0
REFRESH_BUFFER_TIME = 10  # seconds before making consecutive refreshes


def check_lock():
    if not os.path.exists(lock_file):
        return False

    try:
        with portalocker.Lock(lock_file, "r", timeout=0.1) as lock_file_handle:
            data = json.load(lock_file_handle)
            return data["status"] == "locked"
    except (portalocker.LockException, FileNotFoundError, json.JSONDecodeError):
        return False


def refresh():
    while check_lock():
        # print("process locked.. sleeping")
        time.sleep(2)

    global last_refreshed_on
    while int(time.time()) - last_refreshed_on < REFRESH_BUFFER_TIME:
        # print(f"waiting {REFRESH_BUFFER_TIME} secs before the next refresh")
        time.sleep(2)

    # print("Refreshing...")
    last_refreshed_on = int(time.time())
    with portalocker.Lock(target_file, "w") as f:
        f.write(f"SAVE_STATE = {random.randint(1, 1000)}")
    return True


def refresh_worker():
    while True:
        refresh_queue.get()
        refresh()


@app.route("/refresh", methods=["POST"])
def trigger_refresh():
    if refresh_queue.empty():
        refresh_queue.put(True)
        return jsonify({"success": True, "message": "Refresh request queued"})
    else:
        return jsonify({"success": False, "message": "Refresh already queued"})


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200


def run_flask():
    # disabling flask's default logger
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)

    # disabling the output stream
    cli = sys.modules["flask.cli"]
    cli.show_server_banner = lambda *x: None

    app.run(host="0.0.0.0", port=REFRESH_PROCESS_PORT)


def main():
    # flask server
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()

    # refresh worker
    refresh_thread = threading.Thread(target=refresh_worker)
    refresh_thread.daemon = True
    refresh_thread.start()

    # running the main thread as long as the flask server is active
    flask_thread.join()


if __name__ == "__main__":
    main()
