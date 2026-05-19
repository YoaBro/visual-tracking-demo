from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

import config


@dataclass
class MatchResult:
    bbox: Optional[Tuple[int, int, int, int]]
    good_matches: int
    inliers: int


def create_orb() -> cv2.ORB:
    return cv2.ORB_create(
        nfeatures=config.ORB_N_FEATURES,
        scaleFactor=config.ORB_SCALE_FACTOR,
        nlevels=config.ORB_N_LEVELS,
        edgeThreshold=config.ORB_EDGE_THRESHOLD,
        firstLevel=config.ORB_FIRST_LEVEL,
        WTA_K=config.ORB_WTA_K,
        scoreType=config.ORB_SCORE_TYPE,
        patchSize=config.ORB_PATCH_SIZE,
        fastThreshold=config.ORB_FAST_THRESHOLD,
    )


def extract_orb_features(image: np.ndarray, orb: cv2.ORB):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors


def clamp_bbox(bbox: Sequence[float], frame_shape: Sequence[int]) -> Optional[Tuple[int, int, int, int]]:
    if len(frame_shape) < 2:
        return None

    frame_height = int(frame_shape[0])
    frame_width = int(frame_shape[1])
    x, y, w, h = [int(round(value)) for value in bbox]

    if w <= 0 or h <= 0:
        return None

    x = max(0, min(x, frame_width - 1))
    y = max(0, min(y, frame_height - 1))
    w = min(w, frame_width - x)
    h = min(h, frame_height - y)

    if w <= 0 or h <= 0:
        return None

    return x, y, w, h


def draw_transparent_box(image: np.ndarray, bbox: Optional[Tuple[int, int, int, int]], color, alpha: float = 0.2) -> None:
    if bbox is None:
        return

    x, y, w, h = bbox
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, thickness=-1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness=2)


def draw_status_panel(image: np.ndarray, lines: Sequence[str]) -> None:
    if not lines:
        return

    font = config.STATUS_FONT
    scale = config.STATUS_FONT_SCALE
    thickness = config.STATUS_THICKNESS
    margin_x = config.STATUS_TEXT_ORIGIN[0]
    margin_y = config.STATUS_TEXT_ORIGIN[1]

    widths = []
    heights = []
    baseline_values = []
    for line in lines:
        (width, height), baseline = cv2.getTextSize(line, font, scale, thickness)
        widths.append(width)
        heights.append(height)
        baseline_values.append(baseline)

    panel_width = max(widths) + 20
    panel_height = sum(heights[i] + baseline_values[i] + 8 for i in range(len(lines))) + 10
    overlay = image.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_width, 8 + panel_height), config.TEXT_BG_COLOR, thickness=-1)
    cv2.addWeighted(overlay, config.TEXT_BG_ALPHA, image, 1 - config.TEXT_BG_ALPHA, 0, image)

    y = margin_y
    for index, line in enumerate(lines):
        cv2.putText(image, line, (margin_x, y), font, scale, config.TEXT_COLOR, thickness, cv2.LINE_AA)
        y += heights[index] + baseline_values[index] + 8


def make_match_result(bbox: Optional[Tuple[int, int, int, int]], good_matches: int, inliers: int) -> MatchResult:
    return MatchResult(bbox=bbox, good_matches=good_matches, inliers=inliers)
