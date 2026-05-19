from __future__ import annotations

"""Main application runner for the ROI Tracker demo.

This module contains the entry point and the main live loop that
captures frames from a webcam, shows the GUI, and forwards frames
to the `ROITracker` instance. It also contains small helpers for
opening the camera and selecting a region of interest (ROI).

Only user-facing UI and loop control live here; the tracking logic
is implemented in `tracker.py`.
"""

import time
from typing import Optional, Tuple

import cv2
import numpy as np

import config
from tracker import ROITracker, TrackerState
from utils import draw_status_panel, draw_transparent_box


def open_camera(camera_index: int) -> cv2.VideoCapture:
    """Open a system camera and return a cv2.VideoCapture object.

    We try `CAP_DSHOW` first (works well on Windows). If that fails
    we fall back to the default backend. This helper only opens the
    device; it does not configure resolution or other capture settings.
    """

    capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(camera_index)
    return capture


def select_roi(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Show OpenCV's interactive ROI selector and return the picked box.

    The selector is the small window that lets you drag a rectangle
    with the mouse. After dragging, press SPACE or ENTER to confirm
    the selection, or press `c` to cancel. The returned box is a
    4-tuple `(x, y, w, h)` in pixel coordinates, or `None` when the
    selection was canceled or invalid.
    """

    print("OpenCV ROI selector: drag with the mouse, then press SPACE or ENTER to confirm. Press c to cancel.")
    selection = cv2.selectROI(config.WINDOW_NAME, frame, showCrosshair=True, fromCenter=False)
    if selection is None:
        return None

    x, y, w, h = [int(value) for value in selection]
    if w <= 0 or h <= 0:
        return None

    return x, y, w, h


def build_status_lines(
    state: TrackerState,
    match_count: int,
    inliers: int,
    fps: float,
    tracker_name: str,
    roi_keypoints: int,
) -> list[str]:
    """Return a list of status strings to render in the UI panel.

    This just formats the state and debug numbers (FPS, match counts,
    tracker backend name and current ROI keypoint count) for display.
    """

    lines = [f"STATUS: {state.value}"]
    if tracker_name:
        lines.append(f"TRACKER: {tracker_name}")
    if roi_keypoints:
        lines.append(f"ROI ORB KEYPOINTS: {roi_keypoints}")
    if match_count > 0 or inliers > 0:
        lines.append(f"MATCHES: {match_count}  INLIERS: {inliers}")
    lines.append(f"FPS: {fps:.1f}")
    lines.append("Keys: s=select ROI  r=reset  q=quit")
    return lines


def main() -> None:
    """Main application loop.

    Responsibilities:
    - Open the configured camera.
    - Create an `ROITracker` instance and pass frames to it.
    - Render overlays (bounding box, status panel) and handle key input.

    The function intentionally keeps GUI code here and delegates all
    tracking logic to `ROITracker` in `tracker.py`.
    """

    camera = open_camera(config.CAMERA_INDEX)
    if not camera.isOpened():
        print(f"Could not open camera index {config.CAMERA_INDEX}. Try changing CAMERA_INDEX in config.py.")
        return

    tracker = ROITracker()
    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)

    previous_time = time.perf_counter()
    smoothed_fps = 0.0

    while True:
        ok, frame = camera.read()
        if not ok or frame is None:
            print("Failed to read from webcam.")
            break

        state, bbox, match_result = tracker.update(frame)

        # Compute instantaneous FPS and apply simple exponential smoothing.
        # Smoothing makes the displayed FPS less jumpy; `FPS_SMOOTHING` in
        # config controls how much past measurements affect the shown value.
        current_time = time.perf_counter()
        delta_time = max(current_time - previous_time, 1e-6)
        previous_time = current_time
        fps = 1.0 / delta_time
        if smoothed_fps == 0.0:
            smoothed_fps = fps
        else:
            smoothed_fps = config.FPS_SMOOTHING * smoothed_fps + (1.0 - config.FPS_SMOOTHING) * fps

        # Copy the frame before drawing overlays so the raw `frame` stays
        # available for tracker operations and debugging if needed.
        display_frame = frame.copy()
        if bbox is not None:
            box_color = config.BOX_COLOR_TRACKING
            if state == TrackerState.LOST:
                box_color = config.BOX_COLOR_LOST
            elif state == TrackerState.RE_DETECTED:
                box_color = config.BOX_COLOR_REDETECTED
            draw_transparent_box(display_frame, bbox, box_color, alpha=0.15)
        elif state == TrackerState.NO_TARGET:
            draw_transparent_box(display_frame, None, config.BOX_COLOR_IDLE)

        draw_status_panel(
            display_frame,
            build_status_lines(
                state,
                match_result.good_matches,
                match_result.inliers,
                smoothed_fps,
                tracker.tracker_name,
                tracker.last_init_keypoints,
            ),
        )

        # Show help text when there is no active target to guide beginners.
        if state == TrackerState.NO_TARGET:
            cv2.putText(
                display_frame,
                "Press s to select an ROI",
                (10, display_frame.shape[0] - 15),
                config.STATUS_FONT,
                0.65,
                config.TEXT_COLOR,
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(config.WINDOW_NAME, display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("r"):
            tracker.reset()
            continue
        if key == ord("s"):
            selected_bbox = select_roi(frame)
            if selected_bbox is None:
                print("ROI selection canceled or invalid. Drag a box, then press SPACE or ENTER to confirm.")
                continue
            if not tracker.initialize_from_roi(frame, selected_bbox):
                # show an on-screen message so the user notices the failure
                reason = getattr(tracker, "last_init_message", "ROI initialization failed")
                print(f"ROI initialization failed: {reason}")
                # display the message for ~1.5 seconds
                display_msg = f"Init failed: {reason}"
                temp_frame = frame.copy()
                cv2.putText(
                    temp_frame,
                    display_msg,
                    (10, int(temp_frame.shape[0] * 0.5)),
                    config.STATUS_FONT,
                    0.9,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(config.WINDOW_NAME, temp_frame)
                cv2.waitKey(1500)

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
