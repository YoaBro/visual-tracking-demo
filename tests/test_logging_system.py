import os
import tempfile
import unittest

from logging_system import EventLogger


class TestEventLogger(unittest.TestCase):
    def test_log_event_and_recent_events(self):
        logger = EventLogger(max_events=5)
        logger.log_event("app_started")
        logger.log_event("roi_selected", "bbox=(1,2,3,4)")
        lines = logger.get_recent_events()
        self.assertEqual(len(lines), 2)
        self.assertTrue("app_started" in lines[0])
        self.assertTrue("roi_selected" in lines[1])

    def test_log_event_max_events(self):
        logger = EventLogger(max_events=2)
        logger.log_event("event_1")
        logger.log_event("event_2")
        logger.log_event("event_3")
        lines = logger.get_recent_events(5)
        self.assertEqual(len(lines), 2)
        self.assertTrue("event_2" in lines[0])
        self.assertTrue("event_3" in lines[1])

    def test_save_to_file(self):
        logger = EventLogger()
        logger.log_event("app_started")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "session.txt")
            success = logger.save_to_file(path)
            self.assertTrue(success)
            with open(path, "r", encoding="utf-8") as file_handle:
                contents = file_handle.read()
            self.assertIn("app_started", contents)


if __name__ == "__main__":
    unittest.main()
