import json
import os
import threading

from .config import settings

_lock = threading.Lock()


def log_event(event: dict) -> None:
    line = json.dumps(event, default=str)
    with _lock:
        with open(settings.LOG_PATH, "a") as f:
            f.write(line + "\n")


def read_all_events() -> list[dict]:
    if not os.path.exists(settings.LOG_PATH):
        return []
    with open(settings.LOG_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]