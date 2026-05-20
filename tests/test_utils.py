import unittest

import numpy as np

from utils import clamp_bbox, draw_thumbnail_pair, make_match_result


class TestUtils(unittest.TestCase):
    def test_clamp_bbox_returns_none_for_invalid_shape(self):
        self.assertIsNone(clamp_bbox((10, 10, 20, 20), ()))

    def test_clamp_bbox_clamps_to_frame(self):
        bbox = clamp_bbox((-10, -5, 100, 80), (60, 40, 3))
        self.assertEqual(bbox, (0, 0, 40, 60))

    def test_clamp_bbox_rejects_non_positive_size(self):
        self.assertIsNone(clamp_bbox((10, 10, 0, 5), (100, 100, 3)))
        self.assertIsNone(clamp_bbox((10, 10, 5, -1), (100, 100, 3)))

    def test_make_match_result(self):
        result = make_match_result((1, 2, 3, 4), 10, 7)
        self.assertEqual(result.bbox, (1, 2, 3, 4))
        self.assertEqual(result.good_matches, 10)
        self.assertEqual(result.inliers, 7)

    def test_draw_thumbnail_pair_no_error(self):
        canvas = np.zeros((200, 300, 3), dtype=np.uint8)
        left = np.zeros((40, 40, 3), dtype=np.uint8)
        right = np.zeros((40, 40, 3), dtype=np.uint8)
        bottom_y = draw_thumbnail_pair(canvas, left, right, (10, 10), "ROI", "Current")
        self.assertGreater(bottom_y, 0)


if __name__ == "__main__":
    unittest.main()
