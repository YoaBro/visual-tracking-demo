from __future__ import annotations

"""Simple event logging utilities for the ROI tracker demo."""

from collections import deque
from dataclasses import dataclass
from datetime import datetime
import os
from typing import Deque, List


@dataclass
class LogEvent:
    """Represents one timestamped event in the session log."""

    timestamp: str
    event_type: str
    details: str


class EventLogger:
    """Collects session events and optionally persists them to disk."""

    def __init__(self, max_events: int = 100) -> None:
        self._events: Deque[LogEvent] = deque(maxlen=max_events)

    def log_event(self, event_type: str, details: str = "") -> None:
        """Append a new event with a current timestamp."""

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._events.append(LogEvent(timestamp=timestamp, event_type=event_type, details=details))

    def get_recent_events(self, count: int = 8) -> List[str]:
        """Return the most recent events formatted for display."""

        recent = list(self._events)[-count:]
        lines: List[str] = []
        for event in recent:
            if event.details:
                lines.append(f"{event.timestamp} {event.event_type}: {event.details}")
            else:
                lines.append(f"{event.timestamp} {event.event_type}")
        return lines

    def save_to_file(self, filepath: str) -> bool:
        """Write all events to a log file. Returns True on success."""

        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        try:
            with open(filepath, "w", encoding="utf-8") as file_handle:
                for event in self._events:
                    if event.details:
                        file_handle.write(f"{event.timestamp} {event.event_type}: {event.details}\n")
                    else:
                        file_handle.write(f"{event.timestamp} {event.event_type}\n")
            return True
        except OSError:
            return False
