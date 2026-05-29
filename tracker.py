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
    - `SUSPECT`: CSRT ok but visual validation weak (Phase 2); between TRACKING and LOST.
    - `LOST`: tracker failed for the current frame.
    - `RE_DETECTED`: a re-detection produced a new ROI and the tracker
      was rebuilt.
    """

    NO_TARGET = "NO_TARGET"
    TRACKING = "TRACKING"
    SUSPECT = "SUSPECT"
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
        # Remember last valid bbox for spatial gating during re-detection
        self.last_valid_bbox: Optional[Tuple[int, int, int, int]] = None
        # Last confirmed bbox: only updated when TRACKING passes visual validation
        # (Phase 3: protects from false tracking poisoning the re-detection state)
        self.last_confirmed_bbox: Optional[Tuple[int, int, int, int]] = None
        # Pending re-detection candidate confirmation
        self.pending_redetect_bbox: Optional[Tuple[int, int, int, int]] = None
        self.pending_redetect_count: int = 0
        
        # Tracking validation state (Phase 1–3): visual validation of current bbox
        # against reference template to detect tracker drift during TRACKING state.
        self.frame_since_last_track_validation: int = 0  # Counter for every-N-frames check
        self.track_validation_failed_count: int = 0      # Cumulative failed validations
        self.last_track_validation_matches: int = 0      # Debug: matches from last validation
        self.last_track_validation_inliers: int = 0      # Debug: inliers from last validation
        self.last_track_validation_ratio: float = 0.0    # Debug: inlier ratio from validation
        self.last_track_valid_confidence: bool = True    # Debug: result of last validation
        # Detailed per-match points for visualization: list of
        # (template_pt_x, template_pt_y, current_pt_x, current_pt_y, is_inlier)
        self.last_track_validation_match_points: list = []
        # SUSPECT state duration: track how long we've been in SUSPECT
        self.suspect_frame_count: int = 0

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
        self.last_valid_bbox = None
        self.last_confirmed_bbox = None
        self.pending_redetect_bbox = None
        self.pending_redetect_count = 0
        self.frame_since_last_track_validation = 0
        self.track_validation_failed_count = 0
        self.last_track_validation_matches = 0
        self.last_track_validation_inliers = 0
        self.last_track_validation_ratio = 0.0
        self.last_track_valid_confidence = True
        self.last_track_validation_match_points = []
        self.suspect_frame_count = 0

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
        self.last_valid_bbox = valid_bbox
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
        self.last_confirmed_bbox = valid_bbox
        self.pending_redetect_bbox = None
        self.pending_redetect_count = 0
        self.track_validation_failed_count = 0
        self.suspect_frame_count = 0
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
                    self.last_valid_bbox = clamped_bbox
                    
                    # Phase 1: Tracking validation debug (visual validation against reference).
                    # Every N frames during TRACKING, validate the current bbox against the
                    # reference template to detect tracker drift. Store results but don't
                    # change state yet (behavior change deferred to Phase 3).
                    self.frame_since_last_track_validation += 1
                    if self.frame_since_last_track_validation >= config.TRACK_VALIDATE_EVERY_N_FRAMES:
                        is_valid, matches, inliers, ratio = self._validate_tracking(frame, clamped_bbox)
                        self.last_track_validation_matches = matches
                        self.last_track_validation_inliers = inliers
                        self.last_track_validation_ratio = ratio
                        self.last_track_valid_confidence = is_valid
                        
                        # Phase 3: State transition logic based on validation results.
                        if is_valid:
                            # Validation passed: confirmed tracking.
                            self.track_validation_failed_count = 0
                            self.suspect_frame_count = 0
                            self.last_confirmed_bbox = clamped_bbox
                            self.state = TrackerState.TRACKING
                        else:
                            # Validation failed: increment failure counter.
                            self.track_validation_failed_count += 1
                            
                            # Check if we should transition to SUSPECT.
                            if self.track_validation_failed_count >= config.TRACK_MAX_FAILED_VALIDATIONS:
                                if self.state == TrackerState.TRACKING:
                                    # Transition TRACKING -> SUSPECT: visual validation weak.
                                    self.state = TrackerState.SUSPECT
                                    self.suspect_frame_count = 0
                                elif self.state == TrackerState.SUSPECT:
                                    # Already in SUSPECT, increment duration counter.
                                    self.suspect_frame_count += 1
                                    # After a few more failures in SUSPECT, go to LOST.
                                    if self.suspect_frame_count >= config.TRACK_MAX_FAILED_VALIDATIONS:
                                        self.state = TrackerState.LOST
                        
                        self.frame_since_last_track_validation = 0
                    
                    self.state = TrackerState.TRACKING
                    self.last_match_result = make_match_result(clamped_bbox, 0, 0)
                    return self.state, clamped_bbox, self.last_match_result

        # Tracker failed for this frame - try redetection
        # If in SUSPECT, transition to LOST on CSRT failure.
        self.state = TrackerState.LOST
        match_result = self._attempt_redetection(frame)
        self.last_match_result = match_result
        confirmed_bbox = self._confirm_redetection_candidate(match_result.bbox)
        if confirmed_bbox is not None:
            self.current_bbox = confirmed_bbox
            self.last_valid_bbox = confirmed_bbox
            self.last_confirmed_bbox = confirmed_bbox
            self.tracker, self.tracker_name = self._init_tracker(
                frame, tuple(int(value) for value in confirmed_bbox)
            )
            if self.tracker is not None:
                self.state = TrackerState.RE_DETECTED
                self.track_validation_failed_count = 0
                self.suspect_frame_count = 0
                confirmed_result = make_match_result(confirmed_bbox, match_result.good_matches, match_result.inliers)
                return self.state, confirmed_bbox, confirmed_result

        if match_result.bbox is not None:
            match_result = make_match_result(None, match_result.good_matches, match_result.inliers)
            self.last_match_result = match_result

        return self.state, None, match_result

    def _bbox_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x, y, w, h = bbox
        return x + (w / 2.0), y + (h / 2.0)

    def _bbox_area(self, bbox: Tuple[int, int, int, int]) -> float:
        return float(max(bbox[2], 0) * max(bbox[3], 0))

    def _bbox_iou(
        self, first: Tuple[int, int, int, int], second: Tuple[int, int, int, int]
    ) -> float:
        ax, ay, aw, ah = first
        bx, by, bw, bh = second
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh

        inter_x1 = max(ax, bx)
        inter_y1 = max(ay, by)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0
        union_area = (aw * ah) + (bw * bh) - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def _is_candidate_reasonable(self, bbox: Tuple[int, int, int, int], inliers: int, matches: int) -> bool:
        if self.roi_model is None:
            return False

        if matches <= 0:
            return False

        inlier_ratio = inliers / max(matches, 1)
        if inlier_ratio < config.MIN_HOMOGRAPHY_INLIER_RATIO:
            return False

        template_w, template_h = self.roi_model.template_size
        template_area = float(template_w * template_h)
        candidate_area = self._bbox_area(bbox)
        if template_area <= 0 or candidate_area <= 0:
            return False

        area_ratio = candidate_area / template_area
        if area_ratio < config.MIN_REDETECT_AREA_RATIO or area_ratio > config.MAX_REDETECT_AREA_RATIO:
            return False

        template_ar = template_w / max(template_h, 1)
        candidate_ar = bbox[2] / max(bbox[3], 1)
        ar_ratio = max(candidate_ar / template_ar, template_ar / candidate_ar)
        if ar_ratio > (1.0 + config.MAX_REDETECT_ASPECT_RATIO_DELTA):
            return False

        if self.last_valid_bbox is not None:
            last_cx, last_cy = self._bbox_center(self.last_valid_bbox)
            cand_cx, cand_cy = self._bbox_center(bbox)
            distance = float(np.hypot(cand_cx - last_cx, cand_cy - last_cy))
            ref_size = float(max(template_w, template_h))
            if ref_size > 0:
                if distance / ref_size > config.MAX_REDETECT_CENTER_DISTANCE_RATIO:
                    return False

        return True

    def _confirm_redetection_candidate(
        self, bbox: Optional[Tuple[int, int, int, int]]
    ) -> Optional[Tuple[int, int, int, int]]:
        if bbox is None or self.last_match_result is None:
            self.pending_redetect_bbox = None
            self.pending_redetect_count = 0
            return None

        if not self._is_candidate_reasonable(bbox, self.last_match_result.inliers, self.last_match_result.good_matches):
            self.pending_redetect_bbox = None
            self.pending_redetect_count = 0
            return None

        if self.pending_redetect_bbox is None:
            self.pending_redetect_bbox = bbox
            self.pending_redetect_count = 1
            return None

        iou = self._bbox_iou(self.pending_redetect_bbox, bbox)
        if iou >= 0.3:
            self.pending_redetect_count += 1
        else:
            self.pending_redetect_bbox = bbox
            self.pending_redetect_count = 1

        if self.pending_redetect_count >= config.REDETECT_CONFIRM_FRAMES:
            confirmed = self.pending_redetect_bbox
            self.pending_redetect_bbox = None
            self.pending_redetect_count = 0
            return confirmed

        return None

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

    def _validate_tracking(
        self, frame: np.ndarray, current_bbox: Tuple[int, int, int, int]
    ) -> Tuple[bool, int, int, float]:
        """Validate the current tracked bbox against the reference template.

        This method compares the current bbox (from CSRT tracker output) to the
        original reference ROI using ORB feature matching. It helps detect
        tracker drift onto wrong regions during TRACKING state.

        Phase 1–3 improvement: visual validation during tracking, not just re-detection.

        Args:
            frame: The current frame (BGR image).
            current_bbox: The current bounding box from CSRT tracker (x, y, w, h).

        Returns:
            Tuple of (is_valid, matches, inliers, inlier_ratio):
            - is_valid (bool): True if current bbox passes validation.
            - matches (int): Number of good feature matches found.
            - inliers (int): Number of RANSAC inliers.
            - inlier_ratio (float): inliers / matches ratio, or 0.0 if no matches.
        """

        if self.roi_model is None:
            return False, 0, 0, 0.0

        # Extract ROI crop from current frame using the current bbox.
        x, y, w, h = current_bbox
        if w <= 0 or h <= 0 or y + h > frame.shape[0] or x + w > frame.shape[1]:
            return False, 0, 0, 0.0

        current_roi = frame[y : y + h, x : x + w].copy()
        if current_roi.size == 0:
            return False, 0, 0, 0.0

        # Extract ORB features from the current bbox.
        current_keypoints, current_descriptors = extract_orb_features(current_roi, self.orb)
        if current_descriptors is None or len(current_keypoints) < 4:
            return False, 0, 0, 0.0

        # Match current crop descriptors against template descriptors.
        matches = self.matcher.knnMatch(self.roi_model.template_descriptors, current_descriptors, k=2)
        good_matches = []
        for pair in matches:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance < config.RATIO_TEST * second.distance:
                good_matches.append(first)

        if len(good_matches) < config.TRACK_MIN_MATCHES:
            # clear any previous match points
            self.last_track_validation_match_points = []
            return False, len(good_matches), 0, 0.0

        # Compute homography with RANSAC to validate spatial consistency.
        template_points = np.float32(
            [self.roi_model.template_keypoints[match.queryIdx].pt for match in good_matches]
        ).reshape(-1, 1, 2)
        current_points = np.float32(
            [current_keypoints[match.trainIdx].pt for match in good_matches]
        ).reshape(-1, 1, 2)

        homography, inlier_mask = cv2.findHomography(template_points, current_points, cv2.RANSAC, 5.0)
        if homography is None or inlier_mask is None:
            self.last_track_validation_match_points = []
            return False, len(good_matches), 0, 0.0

        inliers = int(inlier_mask.ravel().sum())
        inlier_ratio = inliers / max(len(good_matches), 1)
        # Build detailed per-match list for visualization. Note: template pts
        # are relative to the template image, current pts are relative to the
        # cropped `current_roi` (we'll add bbox offsets when drawing in the UI).
        match_points = []
        for i, match in enumerate(good_matches):
            tpl_pt = self.roi_model.template_keypoints[match.queryIdx].pt
            cur_pt = current_keypoints[match.trainIdx].pt
            is_inlier = bool(inlier_mask.ravel()[i])
            match_points.append((float(tpl_pt[0]), float(tpl_pt[1]), float(cur_pt[0]), float(cur_pt[1]), is_inlier))
        self.last_track_validation_match_points = match_points

        # Validation passes if both inlier count and ratio thresholds are met.
        is_valid = (
            inliers >= config.TRACK_MIN_INLIERS
            and inlier_ratio >= config.TRACK_MIN_INLIER_RATIO
        )
        return is_valid, len(good_matches), inliers, inlier_ratio

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
