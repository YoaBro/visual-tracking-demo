from __future__ import annotations

"""UI state management for Learning Mode, help, and log overlays."""

from typing import Optional

import config
from tracker import TrackerState


class UIManager:
    """Keeps track of UI toggle state for the demo."""

    def __init__(self) -> None:
        # These flags are read by main.py to decide which overlays to render.
        self.learning_mode = False
        self.show_help = False
        self.show_log = False

    def toggle_learning_mode(self) -> bool:
        # Learning Mode enables extra visualizations in the side panel.
        self.learning_mode = not self.learning_mode
        return self.learning_mode

    def toggle_help_panel(self) -> bool:
        # Help panel is a short text overlay about the current state.
        self.show_help = not self.show_help
        return self.show_help

    def toggle_log_view(self) -> bool:
        # Log view shows recent events captured by EventLogger.
        self.show_log = not self.show_log
        return self.show_log


def get_help_text(state: Optional[TrackerState]) -> str:
    """Return a short help description for the current tracker state."""

    if state is None:
        return ""
    # Convert enum values to the string keys stored in config.HELP_TEXT.
    state_key = state.value if isinstance(state, TrackerState) else str(state)
    return config.HELP_TEXT.get(state_key, "")
