import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from screenshot_utils import save_screenshot


class TestScreenshotUtils(unittest.TestCase):
    @patch("screenshot_utils.cv2.imwrite", return_value=True)
    def test_save_screenshot_success(self, mock_imwrite):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp_dir:
            success, message = save_screenshot(image, directory=temp_dir)
            self.assertTrue(success)
            self.assertTrue(message.startswith(f"{temp_dir}/capture_"))
            self.assertTrue(message.endswith(".png"))
            self.assertTrue(os.path.isdir(temp_dir))

    def test_save_screenshot_empty_image(self):
        image = np.zeros((0, 0, 3), dtype=np.uint8)
        success, message = save_screenshot(image, directory="captures")
        self.assertFalse(success)
        self.assertEqual(message, "No image data")


if __name__ == "__main__":
    unittest.main()
