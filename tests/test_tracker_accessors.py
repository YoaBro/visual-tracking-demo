import unittest
import numpy as np

from tracker import ROITracker


class TestTrackerAccessors(unittest.TestCase):
    def test_roi_thumbnail_initially_none(self):
        t = ROITracker()
        self.assertIsNone(t.get_roi_thumbnail())

    def test_match_confidence_initially_zero(self):
        t = ROITracker()
        self.assertEqual(t.get_current_match_confidence(), 0)

    def test_roi_thumbnail_after_set(self):
        t = ROITracker()
        # create a fake roi model
        img = np.zeros((20, 30, 3), dtype=np.uint8)
        t.roi_model = type("X", (), {"template_image": img, "template_keypoints": [], "template_descriptors": None, "template_size": (30,20)})()
        thumb = t.get_roi_thumbnail()
        self.assertIsNotNone(thumb)
        self.assertEqual(thumb.shape[0], 20)
        self.assertEqual(thumb.shape[1], 30)


if __name__ == "__main__":
    unittest.main()
