from __future__ import annotations

"""Main application runner for the ROI Tracker demo.

This module contains the entry point and the main live loop that
captures frames from a webcam, shows the GUI, and forwards frames
to the `ROITracker` instance. It also contains small helpers for
opening the camera and selecting a region of interest (ROI).

Only user-facing UI and loop control live here; the tracking logic
is implemented in `tracker.py`.
"""

from datetime import datetime
import time
from typing import Optional, Tuple

import cv2
import numpy as np

import config
from logging_system import EventLogger
from screenshot_utils import save_screenshot
from tracker import ROITracker, TrackerState
from ui_manager import UIManager, get_help_text
from utils import (
    draw_event_log_panel,
    draw_help_panel,
    draw_side_panel,
    draw_thumbnail_pair,
    draw_transparent_box,
)


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
    learning_mode: bool,
    show_help: bool,
    show_log: bool,
    redetection_reason: Optional[str] = None,
    # Tracking-validation diagnostics (optional)
    track_matches: int = 0,
    track_inliers: int = 0,
    track_ratio: float = 0.0,
    track_valid: bool = False,
    track_fail_count: int = 0,
) -> list[str]:
    """Return a list of status strings to render in the UI panel.

    This just formats the state and debug numbers (FPS, match counts,
    tracker backend name and current ROI keypoint count) for display.
    """

    lines = [f"STATE: {state.value} [?]"]
    lines.append(f"TRACKER: {tracker_name or '-'}")
    lines.append(f"ROI KP: {roi_keypoints}")
    lines.append("")
    lines.append(f"MATCHES: {match_count}  INLIERS: {inliers}")
    lines.append(f"FPS: {fps:.1f}")
    lines.append("")
    # Show per-frame tracking validation diagnostics when a target exists.
    if state != TrackerState.NO_TARGET:
        lines.append(f"TRACK_VALID: {'YES' if track_valid else 'NO'}  FAILS: {track_fail_count}")
        lines.append(f"TV MATCHES: {track_matches}  INLIERS: {track_inliers}  RATIO: {track_ratio:.2f}")
        lines.append("")
    lines.append(f"LEARNING: {'ON' if learning_mode else 'OFF'} [l]")
    lines.append(f"HELP: {'ON' if show_help else 'OFF'} [h]")
    lines.append(f"LOG: {'ON' if show_log else 'OFF'} [g]")
    if learning_mode:
        lines.append("")
        lines.append(f"MIN MATCHES: {config.MIN_GOOD_MATCHES}")
        lines.append(f"MIN INLIERS: {config.MIN_HOMOGRAPHY_INLIERS}")
        if redetection_reason:
            lines.append(f"REDETECT: {redetection_reason}")
    lines.append("")
    lines.append("Keys: s=select r=reset p=snap")
    lines.append("l=learn h=help g=log q=quit")
    return lines


def describe_redetection_failure(match_count: int, inliers: int) -> str:
    """Return a short reason string for a failed re-detection attempt."""

    if match_count < config.MIN_GOOD_MATCHES:
        return f"Too few matches ({match_count}/{config.MIN_GOOD_MATCHES})"
    if inliers < config.MIN_HOMOGRAPHY_INLIERS:
        return f"Too few inliers ({inliers}/{config.MIN_HOMOGRAPHY_INLIERS})"
    return "Homography failed"


def make_placeholder_thumbnail(size: Tuple[int, int], label: str) -> np.ndarray:
    """Create a simple placeholder thumbnail with centered text."""

    width, height = size
    placeholder = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(placeholder, (0, 0), (width - 1, height - 1), config.TEXT_COLOR, thickness=1)
    (text_width, text_height), baseline = cv2.getTextSize(
        label, config.STATUS_FONT, config.PANEL_FONT_SCALE, config.PANEL_THICKNESS
    )
    text_x = max(4, (width - text_width) // 2)
    text_y = max(text_height + 4, (height + text_height) // 2)
    cv2.putText(
        placeholder,
        label,
        (text_x, text_y),
        config.STATUS_FONT,
        config.PANEL_FONT_SCALE,
        config.TEXT_COLOR,
        config.PANEL_THICKNESS,
        cv2.LINE_AA,
    )
    return placeholder


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
    ui_manager = UIManager()
    event_logger = EventLogger()
    event_logger.log_event("app_started")
    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)

    previous_time = time.perf_counter()
    smoothed_fps = 0.0
    last_state = tracker.state
    toast_message = ""
    toast_expires_at = 0.0
    last_thumbnail_status = None

    while True:
        ok, frame = camera.read()
        if not ok or frame is None:
            print("Failed to read from webcam.")
            break

        state, bbox, match_result = tracker.update(frame)
        if state != last_state:
            if state == TrackerState.LOST:
                event_logger.log_event("tracking_lost")
                event_logger.log_event(
                    "redetection_attempt",
                    f"matches={match_result.good_matches} inliers={match_result.inliers}",
                )
                if match_result.bbox is None:
                    reason = describe_redetection_failure(match_result.good_matches, match_result.inliers)
                    event_logger.log_event("redetection_failure", reason)
            elif state == TrackerState.RE_DETECTED:
                if last_state == TrackerState.TRACKING:
                    event_logger.log_event("tracking_lost")
                event_logger.log_event(
                    "redetection_attempt",
                    f"matches={match_result.good_matches} inliers={match_result.inliers}",
                )
                event_logger.log_event(
                    "redetection_success",
                    f"matches={match_result.good_matches} inliers={match_result.inliers}",
                )
            last_state = state

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
            if state == TrackerState.SUSPECT:
                box_color = config.BOX_COLOR_SUSPECT
            elif state == TrackerState.LOST:
                box_color = config.BOX_COLOR_LOST
            elif state == TrackerState.RE_DETECTED:
                box_color = config.BOX_COLOR_REDETECTED
            draw_transparent_box(display_frame, bbox, box_color, alpha=0.15)
        elif state == TrackerState.NO_TARGET:
            draw_transparent_box(display_frame, None, config.BOX_COLOR_IDLE)

        panel_width = config.SIDE_PANEL_WIDTH
        panel_padding = config.SIDE_PANEL_PADDING
        height, width = display_frame.shape[:2]
        display_canvas = np.zeros((height, width + panel_width, 3), dtype=display_frame.dtype)
        display_canvas[:, :width] = display_frame

        redetection_reason = None
        if ui_manager.learning_mode and state == TrackerState.LOST and match_result.bbox is None:
            redetection_reason = describe_redetection_failure(match_result.good_matches, match_result.inliers)

        status_lines = build_status_lines(
            state,
            match_result.good_matches,
            match_result.inliers,
            smoothed_fps,
            tracker.tracker_name,
            tracker.last_init_keypoints,
            ui_manager.learning_mode,
            ui_manager.show_help,
            ui_manager.show_log,
            redetection_reason,
            tracker.last_track_validation_matches,
            tracker.last_track_validation_inliers,
            tracker.last_track_validation_ratio,
            tracker.last_track_valid_confidence,
            tracker.track_validation_failed_count,
        )

        panel_rect = (width, 0, panel_width, height)
        text_bottom = draw_side_panel(display_canvas, status_lines, panel_rect)

        if ui_manager.learning_mode:
            roi_thumbnail = tracker.get_roi_thumbnail()
            roi_present = roi_thumbnail is not None and roi_thumbnail.size > 0
            if not roi_present:
                roi_thumbnail = make_placeholder_thumbnail(config.THUMBNAIL_SIZE, "No ROI")

            match_thumbnail = None
            match_present = False
            if bbox is not None:
                x, y, w, h = bbox
                if w > 0 and h > 0:
                    x2 = x + w
                    y2 = y + h
                    if 0 <= x < width and 0 <= y < height and x2 <= width and y2 <= height:
                        match_thumbnail = display_frame[y:y2, x:x2].copy()
                        match_present = match_thumbnail.size > 0
            if not match_present:
                match_thumbnail = make_placeholder_thumbnail(config.THUMBNAIL_SIZE, "No match")

            # Draw matching keypoints on the thumbnails if available.
            match_points = getattr(tracker, "last_track_validation_match_points", None)
            if match_points and tracker.roi_model is not None:
                # Draw on the original ROI thumbnail (template coords).
                try:
                    for tpl_x, tpl_y, cur_x, cur_y, is_inlier in match_points:
                        color = (0, 255, 0) if is_inlier else (0, 128, 255)
                        cv2.circle(roi_thumbnail, (int(round(tpl_x)), int(round(tpl_y))), 3, color, thickness=-1)
                    # Draw on the match thumbnail (current crop coords)
                    for tpl_x, tpl_y, cur_x, cur_y, is_inlier in match_points:
                        color = (0, 255, 0) if is_inlier else (0, 128, 255)
                        cv2.circle(match_thumbnail, (int(round(cur_x)), int(round(cur_y))), 3, color, thickness=-1)
                except Exception:
                    # Drawing should never crash the main loop; ignore on error.
                    pass

            thumbnail_status = (roi_present, match_present)
            if thumbnail_status != last_thumbnail_status:
                event_logger.log_event(
                    "thumbnail_draw_attempt",
                    f"roi={'yes' if roi_present else 'no'}, match={'yes' if match_present else 'no'}",
                )
                last_thumbnail_status = thumbnail_status

            thumb_origin_x = width + panel_padding
            thumb_block_height = config.THUMBNAIL_SIZE[1] + 30
            max_thumb_y = height - panel_padding - thumb_block_height
            if ui_manager.show_log:
                max_thumb_y = min(max_thumb_y, height - panel_padding - config.LOG_PANEL_HEIGHT - thumb_block_height)
            thumb_origin_y = min(max(text_bottom + panel_padding, panel_padding + 40), max_thumb_y)
            if max_thumb_y >= panel_padding:
                draw_thumbnail_pair(
                    display_canvas,
                    roi_thumbnail,
                    match_thumbnail,
                    (thumb_origin_x, thumb_origin_y),
                    "ROI",
                    "Current",
                    highlight_right=state in (TrackerState.TRACKING, TrackerState.RE_DETECTED),
                )

        if ui_manager.show_log:
            log_lines = event_logger.get_recent_events()
            log_panel_width = panel_width - (panel_padding * 2)
            log_panel_x = width + panel_padding
            log_panel_y = height - config.LOG_PANEL_HEIGHT - panel_padding
            log_panel_rect = (log_panel_x, log_panel_y, log_panel_width, config.LOG_PANEL_HEIGHT)
            draw_event_log_panel(display_canvas, log_lines, log_panel_rect)

        if ui_manager.show_help:
            help_lines = [f"STATE: {state.value}", get_help_text(state)]
            help_lines.append("ROI = Region of Interest (selected box).")
            help_lines.append("Matches = ORB descriptor pairs.")
            help_lines.append("Inliers = matches that fit homography.")
            help_panel_width = min(config.HELP_PANEL_WIDTH, width - 20)
            help_panel_rect = (
                10,
                height - config.HELP_PANEL_HEIGHT - 10,
                help_panel_width,
                config.HELP_PANEL_HEIGHT,
            )
            draw_help_panel(display_canvas, help_lines, help_panel_rect)

        if toast_message and time.perf_counter() < toast_expires_at:
            (text_width, text_height), baseline = cv2.getTextSize(
                toast_message, config.STATUS_FONT, config.PANEL_FONT_SCALE, config.PANEL_THICKNESS
            )
            box_x = 10
            box_y = 10
            box_width = text_width + 20
            box_height = text_height + baseline + 12
            overlay = display_canvas.copy()
            cv2.rectangle(
                overlay,
                (box_x, box_y),
                (box_x + box_width, box_y + box_height),
                config.TEXT_BG_COLOR,
                thickness=-1,
            )
            cv2.addWeighted(overlay, config.TEXT_BG_ALPHA, display_canvas, 1 - config.TEXT_BG_ALPHA, 0, display_canvas)
            cv2.putText(
                display_canvas,
                toast_message,
                (box_x + 10, box_y + text_height + 8),
                config.STATUS_FONT,
                config.PANEL_FONT_SCALE,
                config.TEXT_COLOR,
                config.PANEL_THICKNESS,
                cv2.LINE_AA,
            )

        # Show help text when there is no active target to guide beginners.
        if state == TrackerState.NO_TARGET:
            cv2.putText(
                display_canvas,
                "Press s to select an ROI",
                (10, display_frame.shape[0] - 15),
                config.STATUS_FONT,
                0.65,
                config.TEXT_COLOR,
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(config.WINDOW_NAME, display_canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            event_logger.log_event("quit")
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            event_logger.save_to_file(f"logs/session_{timestamp}.txt")
            break
        if key == ord("r"):
            tracker.reset()
            event_logger.log_event("reset")
            last_state = tracker.state
            continue
        if key == ord("s"):
            selected_bbox = select_roi(frame)
            if selected_bbox is None:
                print("ROI selection canceled or invalid. Drag a box, then press SPACE or ENTER to confirm.")
                event_logger.log_event("invalid_roi_rejected", "Selection canceled or invalid")
                continue
            event_logger.log_event("roi_selected", f"bbox={selected_bbox}")
            if not tracker.initialize_from_roi(frame, selected_bbox):
                # show an on-screen message so the user notices the failure
                reason = getattr(tracker, "last_init_message", "ROI initialization failed")
                print(f"ROI initialization failed: {reason}")
                event_logger.log_event("invalid_roi_rejected", reason)
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
            else:
                event_logger.log_event("tracker_initialized", tracker.tracker_name or "unknown")

        if key == ord("p"):
            success, message = save_screenshot(display_canvas)
            toast_message = "Screenshot saved" if success else message
            toast_expires_at = time.perf_counter() + 1.5
            if success:
                event_logger.log_event("screenshot_saved", message)

        if key == ord("l"):
            ui_manager.toggle_learning_mode()

        if key == ord("h"):
            ui_manager.toggle_help_panel()

        if key == ord("g"):
            ui_manager.toggle_log_view()

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
