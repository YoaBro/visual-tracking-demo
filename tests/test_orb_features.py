import unittest

import numpy as np

from utils import create_orb, extract_orb_features


class TestOrbFeatures(unittest.TestCase):
    def test_extract_orb_features_returns_keypoints(self):
        orb = create_orb()
        img = np.zeros((120, 160, 3), dtype=np.uint8)
        cv = 255
        img[20:100, 40:120] = (cv, cv, cv)
        keypoints, descriptors = extract_orb_features(img, orb)
        self.assertIsNotNone(keypoints)
        self.assertTrue(isinstance(keypoints, (list, tuple)))
        # On a mostly flat image, descriptors might be None; just ensure no crash.


if __name__ == "__main__":
    unittest.main()
