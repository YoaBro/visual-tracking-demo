from __future__ import annotations

"""Screenshot helper for saving the current display frame."""

from datetime import datetime
import os
from typing import Tuple

import cv2
import numpy as np


def save_screenshot(image: np.ndarray, directory: str = "captures") -> Tuple[bool, str]:
    """Save the provided image into `directory` with a timestamped filename."""

    if image is None or image.size == 0:
        return False, "No image data"

    # Ensure output directory exists, then save a timestamped PNG.
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"capture_{timestamp}.png"
    relative_path = f"{directory}/{filename}"
    file_path = os.path.join(directory, filename)

    # Return (True, relative_path) on success; otherwise a message string.
    if cv2.imwrite(file_path, image):
        return True, relative_path
    return False, "Failed to save screenshot"
