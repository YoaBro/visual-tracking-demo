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

# Status panel and drawing parameters
STATUS_TEXT_ORIGIN = (10, 28)
STATUS_FONT = 0
STATUS_FONT_SCALE = 0.7
STATUS_THICKNESS = 2

# Colors used for the bounding box in BGR format
BOX_COLOR_TRACKING = (0, 220, 0)
BOX_COLOR_LOST = (0, 165, 255)
BOX_COLOR_REDETECTED = (255, 180, 0)
BOX_COLOR_IDLE = (160, 160, 160)

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
