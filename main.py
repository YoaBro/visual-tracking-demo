from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np

import config
from tracker import ROITracker, TrackerState
from utils import draw_status_panel, draw_transparent_box


def open_camera(camera_index: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(camera_index)
    return capture


def select_roi(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
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

        current_time = time.perf_counter()
        delta_time = max(current_time - previous_time, 1e-6)
        previous_time = current_time
        fps = 1.0 / delta_time
        if smoothed_fps == 0.0:
            smoothed_fps = fps
        else:
            smoothed_fps = config.FPS_SMOOTHING * smoothed_fps + (1.0 - config.FPS_SMOOTHING) * fps

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
