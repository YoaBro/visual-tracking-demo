from __future__ import annotations

"""Configuration constants for the ROI Tracker demo.

This file centralizes tuning parameters and display constants so it's
easy to adjust behavior from one place. Values are chosen for a
typical laptop webcam; change them slowly when experimenting.
"""

# Which camera index to open (try 0, 1, 2...) — use detect_cameras.py to probe
CAMERA_INDEX = 1

# GUI window title
WINDOW_NAME = "ROI Tracker Demo"

# Minimum allowed ROI size in pixels. Small ROIs often lack enough features.
MIN_ROI_WIDTH = 40
MIN_ROI_HEIGHT = 40

# Matching / redetection thresholds
# Minimum number of "good" descriptor matches before attempting homography.
MIN_GOOD_MATCHES = 12
# Lowe's ratio test constant for filtering ambiguous descriptor matches.
RATIO_TEST = 0.75
# Minimum number of RANSAC inliers required for a homography to be trusted.
MIN_HOMOGRAPHY_INLIERS = 10
# Minimum inlier ratio (inliers / good matches) required for re-detection.
MIN_HOMOGRAPHY_INLIER_RATIO = 0.35
# Max acceptable area scale change vs. template area during re-detection.
MAX_REDETECT_AREA_RATIO = 2.5
# Min acceptable area scale change vs. template area during re-detection.
MIN_REDETECT_AREA_RATIO = 0.5
# Max acceptable aspect ratio change vs. template aspect ratio.
MAX_REDETECT_ASPECT_RATIO_DELTA = 0.6
# Max acceptable center distance (relative to template size) for re-detection.
MAX_REDETECT_CENTER_DISTANCE_RATIO = 4.0
# Number of consecutive frames required to confirm re-detection.
REDETECT_CONFIRM_FRAMES = 2

# ORB feature detector parameters. Increasing `ORB_N_FEATURES` makes ORB
# detect more features (higher memory and CPU usage). `scaleFactor` and
# `nlevels` control the image pyramid used by ORB.
MAX_TEMPLATE_KEYPOINTS = 800
ORB_N_FEATURES = 1000
ORB_SCALE_FACTOR = 1.2
ORB_N_LEVELS = 8
ORB_EDGE_THRESHOLD = 31
ORB_FIRST_LEVEL = 0
ORB_WTA_K = 2
ORB_SCORE_TYPE = 0
ORB_PATCH_SIZE = 31
ORB_FAST_THRESHOLD = 20

# Tracking validation thresholds used to detect blank/dark frames or
# low-texture ROIs where the tracker is likely to drift.
MIN_TRACK_KEYPOINTS = 8
MIN_TRACK_MEAN_INTENSITY = 25
MIN_TRACK_STDDEV = 10

# Visual validation of tracking against reference template (Phase 1–3 improvements)
# Frequency: validate current bbox every N frames to detect tracker drift early.
TRACK_VALIDATE_EVERY_N_FRAMES = 5
# Minimum descriptor matches required to validate tracking.
TRACK_MIN_MATCHES = 8
# Minimum RANSAC inliers required for tracking validation.
TRACK_MIN_INLIERS = 5
# Minimum inlier ratio (inliers / good matches) for validation.
TRACK_MIN_INLIER_RATIO = 0.35
# Number of consecutive validation failures before entering SUSPECT state.
TRACK_MAX_FAILED_VALIDATIONS = 3

# Status panel and drawing parameters
STATUS_TEXT_ORIGIN = (10, 28)
STATUS_FONT = 0
STATUS_FONT_SCALE = 0.7
STATUS_THICKNESS = 2

# Colors used for the bounding box in BGR format
BOX_COLOR_TRACKING = (0, 220, 0)      # Green: confirmed tracking
BOX_COLOR_SUSPECT = (0, 165, 255)     # Orange: CSRT ok but validation weak (Phase 2)
BOX_COLOR_LOST = (0, 0, 255)          # Red: tracking lost
BOX_COLOR_REDETECTED = (255, 180, 0)  # Blue: re-detected candidate
BOX_COLOR_IDLE = (160, 160, 160)      # Gray: no target selected

TEXT_COLOR = (255, 255, 255)
TEXT_BG_COLOR = (0, 0, 0)
TEXT_BG_ALPHA = 0.55

# FPS smoothing factor in the main loop (0..1). Closer to 1 => smoother value.
FPS_SMOOTHING = 0.9

# Side panel layout settings (pixels)
SIDE_PANEL_WIDTH = 300
SIDE_PANEL_PADDING = 10
PANEL_FONT_SCALE = 0.6
PANEL_THICKNESS = 1
PANEL_LINE_GAP = 6
PANEL_SECTION_GAP = 10
PANEL_BG_ALPHA = 0.75

# Optional overlay panels
HELP_PANEL_WIDTH = 420
HELP_PANEL_HEIGHT = 150
LOG_PANEL_HEIGHT = 140

# Learning Mode thumbnails
THUMBNAIL_SIZE = (120, 90)
THUMBNAIL_GAP = 10

# Short help text shown when the help panel is toggled on.
HELP_TEXT = {
	"NO_TARGET": "No ROI selected. Press 's' to pick a region to track.",
	"TRACKING": "Tracking: the tracker follows the ROI each frame.",
	"LOST": "Lost: tracking failed. Re-detection uses ORB matching.",
	"RE-DETECTED": "Re-detected: a match was found and tracking resumed.",
}
