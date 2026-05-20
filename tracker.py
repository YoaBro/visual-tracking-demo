from __future__ import annotations

"""ROI tracking and redetection machinery.

This module implements `ROITracker`, a thin state machine that wraps an
OpenCV single-object tracker (CSRT/MOSSE/KCF/MIL) for frame-to-frame
tracking and a fallback re-detection pipeline based on ORB feature
matching + homography. The tracker exposes a simple `initialize_from_roi`
and `update` API used by the main loop.

Conceptual flow:
- User selects an ROI -> we compute ORB keypoints/descriptors for the
  ROI and build a template model.
- We initialize an OpenCV tracker for fast real-time updates.
- On each frame we call `tracker.update(frame)`; if the tracker fails
  we run ORB-based re-detection across the whole frame and try to
  rebuild the tracker from the re-detected bounding box.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np

import config
from utils import MatchResult, clamp_bbox, create_orb, extract_orb_features, make_match_result


class TrackerState(str, Enum):
    """Simple tracker state machine values.

    - `NO_TARGET`: no ROI selected.
    - `TRACKING`: tracker is currently following the object.
    - `LOST`: tracker failed for the current frame.
    - `RE_DETECTED`: a re-detection produced a new ROI and the tracker
      was rebuilt.
    """

    NO_TARGET = "NO_TARGET"
    TRACKING = "TRACKING"
    LOST = "LOST"
    RE_DETECTED = "RE-DETECTED"


@dataclass
class ROIModel:
    """Stores template image, keypoints and descriptors for the ROI.

    This is used by the redetection step: descriptors from the template
    are matched against descriptors from the current frame.
    """

    template_image: np.ndarray
    template_keypoints: list
    template_descriptors: Optional[np.ndarray]
    template_size: Tuple[int, int]


class ROITracker:
    """Encapsulates tracking state and redetection logic.

    Public methods:
    - `initialize_from_roi(frame, bbox)` - build template and start tracker
    - `update(frame)` - update tracker and attempt re-detection on failure
    - `reset()` - clear state
    """

    def __init__(self) -> None:
        # Tracker state and CV helpers
        self.state = TrackerState.NO_TARGET
        self.orb = create_orb()
        # BFMatcher with Hamming norm is appropriate for ORB binary descriptors
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        # OpenCV single-object tracker instance (CSRT/MOSSE/KCF/MIL)
        self.tracker = None
        self.tracker_name: str = ""

        # Template / model data used for re-detection
        self.roi_model: Optional[ROIModel] = None
        self.current_bbox: Optional[Tuple[int, int, int, int]] = None
        self.last_match_result = make_match_result(None, 0, 0)
        self.last_init_message: str = ""
        # diagnostic: how many keypoints we found when initializing the ROI
        self.last_init_keypoints: int = 0

    def reset(self) -> None:
        """Clear the tracker and template data, returning to NO_TARGET."""
        self.state = TrackerState.NO_TARGET
        self.tracker = None
        self.tracker_name = ""
        self.roi_model = None
        self.current_bbox = None
        self.last_match_result = make_match_result(None, 0, 0)
        self.last_init_message = ""
        self.last_init_keypoints = 0

    def initialize_from_roi(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """Initialize the ROI template and start an OpenCV tracker.

        Steps:
        1. Clamp and validate the supplied `bbox`.
        2. Extract ORB keypoints/descriptors from the ROI; require a
           minimum number of keypoints to proceed.
        3. Build the `ROIModel` and try to initialize a fast OpenCV
           single-object tracker (CSRT/MOSSE/KCF/MIL). If tracker
           initialization fails we clean up and report failure.
        """

        frame = np.ascontiguousarray(frame)
        valid_bbox = clamp_bbox(bbox, frame.shape)
        if valid_bbox is None:
            self.last_init_message = "Invalid ROI coordinates"
            return False

        x, y, w, h = valid_bbox
        if w < config.MIN_ROI_WIDTH or h < config.MIN_ROI_HEIGHT:
            self.last_init_message = f"ROI too small (min {config.MIN_ROI_WIDTH}x{config.MIN_ROI_HEIGHT})"
            return False

        # Extract template image and compute ORB features for the ROI.
        roi_image = frame[y : y + h, x : x + w].copy()
        template_keypoints, template_descriptors = extract_orb_features(roi_image, self.orb)
        self.last_init_keypoints = len(template_keypoints)
        if template_descriptors is None or len(template_keypoints) < 4:
            self.last_init_message = f"Not enough ORB features in ROI ({len(template_keypoints)} keypoints)"
            return False

        self.roi_model = ROIModel(
            template_image=roi_image,
            template_keypoints=template_keypoints,
            template_descriptors=template_descriptors,
            template_size=(w, h),
        )

        self.current_bbox = valid_bbox
        tracker_bbox = (int(x), int(y), int(w), int(h))
        self.tracker, self.tracker_name = self._init_tracker(frame, tracker_bbox)
        if self.tracker is None:
            self.roi_model = None
            self.current_bbox = None
            self.last_init_message = "All tracker backends failed to initialize"
            return False

        self.state = TrackerState.TRACKING
        self.last_match_result = make_match_result(valid_bbox, 0, 0)
        self.last_init_message = ""
        return True

    def update(self, frame: np.ndarray) -> Tuple[TrackerState, Optional[Tuple[int, int, int, int]], MatchResult]:
        """Update the tracker with the new `frame`.

        Behavior:
        - If the internal tracker reports success, validate the bbox and
          return the `TRACKING` state.
        - If tracking fails, move to `LOST` and attempt ORB-based
          re-detection across the full frame. If re-detection succeeds
          and we can reinitialize the tracker, return `RE_DETECTED`.
        """

        frame = np.ascontiguousarray(frame)
        if self.state == TrackerState.NO_TARGET or self.tracker is None or self.current_bbox is None:
            return self.state, None, self.last_match_result

        ok, tracked_bbox = self.tracker.update(frame)
        if ok:
            clamped_bbox = clamp_bbox(tracked_bbox, frame.shape)
            if clamped_bbox is not None:
                if self._is_tracking_valid(frame, clamped_bbox):
                    self.current_bbox = clamped_bbox
                    self.state = TrackerState.TRACKING
                    self.last_match_result = make_match_result(clamped_bbox, 0, 0)
                    return self.state, clamped_bbox, self.last_match_result

        # Tracker failed for this frame - try redetection
        self.state = TrackerState.LOST
        match_result = self._attempt_redetection(frame)
        self.last_match_result = match_result
        if match_result.bbox is not None:
            self.current_bbox = match_result.bbox
            self.tracker, self.tracker_name = self._init_tracker(
                frame, tuple(int(value) for value in match_result.bbox)
            )
            if self.tracker is not None:
                self.state = TrackerState.RE_DETECTED
                return self.state, match_result.bbox, match_result

        return self.state, None, match_result

    def _is_tracking_valid(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """Basic sanity checks to reduce drift onto blank/dark regions.

        The checks include simple intensity statistics and a minimum
        number of ORB keypoints inside the candidate bbox. This helps
        avoid accepting spurious tracker updates when the camera view
        is covered or the scene is textureless.
        """

        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return False

        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            return False

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        mean, stddev = cv2.meanStdDev(gray)
        mean_val = float(mean[0][0])
        std_val = float(stddev[0][0])

        if mean_val < config.MIN_TRACK_MEAN_INTENSITY and std_val < config.MIN_TRACK_STDDEV:
            return False

        keypoints, _ = extract_orb_features(roi, self.orb)
        if len(keypoints) < config.MIN_TRACK_KEYPOINTS:
            return False

        return True

    def _init_tracker(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]):
        """Try to create and initialize an OpenCV tracker backend.

        Many OpenCV installations expose tracker constructors in either
        the top-level module or the `cv2.legacy` namespace; this method
        attempts common backends and returns the first successfully
        initialized tracker instance and its name.
        """

        tracker_factories = [
            ("CSRT", getattr(cv2, "TrackerCSRT_create", None)),
            ("CSRT", getattr(cv2, "legacy", None) and getattr(cv2.legacy, "TrackerCSRT_create", None)),
            ("MOSSE", getattr(cv2, "TrackerMOSSE_create", None)),
            ("MOSSE", getattr(cv2, "legacy", None) and getattr(cv2.legacy, "TrackerMOSSE_create", None)),
            ("KCF", getattr(cv2, "TrackerKCF_create", None)),
            ("KCF", getattr(cv2, "legacy", None) and getattr(cv2.legacy, "TrackerKCF_create", None)),
            ("MIL", getattr(cv2, "TrackerMIL_create", None)),
            ("MIL", getattr(cv2, "legacy", None) and getattr(cv2.legacy, "TrackerMIL_create", None)),
        ]

        attempted = []
        for name, factory in tracker_factories:
            if factory is None:
                continue
            attempted.append(name)
            try:
                tracker = factory()
            except Exception:
                continue
            if tracker is None:
                continue
            try:
                ok = tracker.init(frame, bbox)
            except Exception:
                ok = False
            if ok:
                return tracker, name

        if attempted:
            self.last_init_message = "Tried trackers: " + ", ".join(attempted)
        return None, ""

    def _attempt_redetection(self, frame: np.ndarray) -> MatchResult:
        """Try to find the ROI in the full `frame` using ORB matching.

        Steps:
        1. Extract ORB descriptors from the full frame.
        2. Match template descriptors to frame descriptors using
           knnMatch + Lowe ratio test to filter ambiguous matches.
        3. If enough good matches are found, compute a homography with
           RANSAC and count inliers to ensure spatial consistency.
        4. If inliers are sufficient, project the template corners to
           the frame and return the bounding box.

        Re-detection can fail for several reasons: too few descriptor
        matches (low texture), many repeated patterns (ambiguous matches),
        or insufficient geometric agreement (homography RANSAC fails).
        """

        if self.roi_model is None:
            return make_match_result(None, 0, 0)

        frame_keypoints, frame_descriptors = extract_orb_features(frame, self.orb)
        if frame_descriptors is None or len(frame_keypoints) < 4:
            return make_match_result(None, 0, 0)

        matches = self.matcher.knnMatch(self.roi_model.template_descriptors, frame_descriptors, k=2)
        good_matches = []
        for pair in matches:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance < config.RATIO_TEST * second.distance:
                good_matches.append(first)

        if len(good_matches) < config.MIN_GOOD_MATCHES:
            return make_match_result(None, len(good_matches), 0)

        template_points = np.float32([self.roi_model.template_keypoints[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
        frame_points = np.float32([frame_keypoints[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)

        homography, inlier_mask = cv2.findHomography(template_points, frame_points, cv2.RANSAC, 5.0)
        if homography is None or inlier_mask is None:
            return make_match_result(None, len(good_matches), 0)

        inliers = int(inlier_mask.ravel().sum())
        if inliers < config.MIN_HOMOGRAPHY_INLIERS:
            return make_match_result(None, len(good_matches), inliers)

        width, height = self.roi_model.template_size
        template_corners = np.float32(
            [[0, 0], [width, 0], [width, height], [0, height]]
        ).reshape(-1, 1, 2)
        projected_corners = cv2.perspectiveTransform(template_corners, homography)

        x_values = projected_corners[:, 0, 0]
        y_values = projected_corners[:, 0, 1]
        x = int(np.floor(np.min(x_values)))
        y = int(np.floor(np.min(y_values)))
        w = int(np.ceil(np.max(x_values) - np.min(x_values)))
        h = int(np.ceil(np.max(y_values) - np.min(y_values)))
        bbox = clamp_bbox((x, y, w, h), frame.shape)
        return make_match_result(bbox, len(good_matches), inliers)

    # Informational accessors for Learning Mode (no logic changes)
    def get_roi_thumbnail(self) -> Optional[np.ndarray]:
        """Return a copy of the original ROI template image, or None."""

        if self.roi_model is None:
            return None
        return self.roi_model.template_image.copy()

    def get_current_match_confidence(self) -> int:
        """Return a simple integer confidence value for the last match.

        We expose the number of inliers from the last re-detection as a
        lightweight confidence indicator for display in Learning Mode.
        """

        return int(self.last_match_result.inliers)
