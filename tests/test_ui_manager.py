import unittest

from tracker import TrackerState
from ui_manager import UIManager, get_help_text


class TestUIManager(unittest.TestCase):
    def test_toggle_learning_mode(self):
        ui = UIManager()
        self.assertFalse(ui.learning_mode)
        self.assertTrue(ui.toggle_learning_mode())
        self.assertFalse(ui.toggle_learning_mode())

    def test_toggle_help_and_log(self):
        ui = UIManager()
        self.assertFalse(ui.show_help)
        self.assertFalse(ui.show_log)
        self.assertTrue(ui.toggle_help_panel())
        self.assertTrue(ui.toggle_log_view())
        self.assertFalse(ui.toggle_help_panel())
        self.assertFalse(ui.toggle_log_view())

    def test_get_help_text(self):
        text = get_help_text(TrackerState.NO_TARGET)
        self.assertIsInstance(text, str)
        self.assertTrue(len(text) > 0)


if __name__ == "__main__":
    unittest.main()
