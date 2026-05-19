import unittest

import numpy as np

import config
from tracker import ROITracker


class TestTrackerValidation(unittest.TestCase):
    def setUp(self):
        self.tracker = ROITracker()

    def test_tracking_invalid_on_dark_frame(self):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        bbox = (10, 10, 50, 50)
        self.assertFalse(self.tracker._is_tracking_valid(frame, bbox))

    def test_tracking_valid_on_textured_frame(self):
        rng = np.random.default_rng(0)
        noise = rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8)
        bbox = (10, 10, 50, 50)
        result = self.tracker._is_tracking_valid(noise, bbox)
        self.assertIsInstance(result, bool)

    def test_tracking_invalid_on_tiny_bbox(self):
        frame = np.full((80, 80, 3), 128, dtype=np.uint8)
        bbox = (10, 10, 0, 10)
        self.assertFalse(self.tracker._is_tracking_valid(frame, bbox))


if __name__ == "__main__":
    unittest.main()
