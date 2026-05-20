from __future__ import annotations

"""UI state management for Learning Mode, help, and log overlays."""

from typing import Optional

import config
from tracker import TrackerState


class UIManager:
    """Keeps track of UI toggle state for the demo."""

    def __init__(self) -> None:
        self.learning_mode = False
        self.show_help = False
        self.show_log = False

    def toggle_learning_mode(self) -> bool:
        self.learning_mode = not self.learning_mode
        return self.learning_mode

    def toggle_help_panel(self) -> bool:
        self.show_help = not self.show_help
        return self.show_help

    def toggle_log_view(self) -> bool:
        self.show_log = not self.show_log
        return self.show_log


def get_help_text(state: Optional[TrackerState]) -> str:
    """Return a short help description for the current tracker state."""

    if state is None:
        return ""
    state_key = state.value if isinstance(state, TrackerState) else str(state)
    return config.HELP_TEXT.get(state_key, "")
