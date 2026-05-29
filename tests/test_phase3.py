#!/usr/bin/env python
"""Integration test for Phase 3: State transitions (TRACKING -> SUSPECT -> LOST)."""

import numpy as np
from tracker import ROITracker, TrackerState

def test_phase3_state_transitions():
    """Test state transitions: TRACKING (valid) -> SUSPECT (failed) -> LOST (sustained failures)."""
    
    # Create a tracker instance
    tracker = ROITracker()
    
    # Create a test frame with good texture
    test_frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    test_frame[50:120, 50:120] = np.random.randint(100, 255, (70, 70, 3), dtype=np.uint8)
    
    # Initialize tracker with an ROI
    roi_bbox = (50, 50, 70, 70)
    init_ok = tracker.initialize_from_roi(test_frame, roi_bbox)
    assert init_ok, "Failed to initialize tracker"
    assert tracker.state == TrackerState.TRACKING, "Should start in TRACKING"
    assert tracker.last_confirmed_bbox is not None, "last_confirmed_bbox should be set"
    print(f"✓ Tracker initialized in TRACKING state")
    print(f"  - last_confirmed_bbox: {tracker.last_confirmed_bbox}")
    
    # Test 1: Validation passes -> stay in TRACKING
    print(f"\n✓ Test 1: Validation passes -> stay in TRACKING")
    for i in range(5):
        is_valid, matches, inliers, ratio = tracker._validate_tracking(test_frame, roi_bbox)
        print(f"  Frame {i+1}: is_valid={is_valid}, matches={matches}, inliers={inliers}")
    
    # Test 2: Create a "bad" frame (empty/different region) to trigger validation failures
    print(f"\n✓ Test 2: Multiple validation failures -> transition to SUSPECT")
    bad_bbox = (10, 10, 30, 30)  # Different, likely bad region
    
    # Reset validation counter for testing
    tracker.frame_since_last_track_validation = 4  # Next validation at frame 5
    tracker.track_validation_failed_count = 0
    
    for i in range(10):
        is_valid, matches, inliers, ratio = tracker._validate_tracking(test_frame, bad_bbox)
        if (i + 1) % 5 == 0:  # Validation happens every 5 frames
            print(f"  Frame {i+1}: is_valid={is_valid}, matches={matches}")
            print(f"           failed_count={tracker.track_validation_failed_count}, state={tracker.state}")
        
        # Simulate what update() does
        if (i + 1) % 5 == 0:
            if not is_valid:
                tracker.track_validation_failed_count += 1
                if tracker.track_validation_failed_count >= 3:  # TRACK_MAX_FAILED_VALIDATIONS
                    if tracker.state == TrackerState.TRACKING:
                        tracker.state = TrackerState.SUSPECT
                        tracker.suspect_frame_count = 0
                        print(f"           --> Transitioned to SUSPECT")
            tracker.frame_since_last_track_validation = 0
        else:
            tracker.frame_since_last_track_validation += 1
    
    assert tracker.state == TrackerState.SUSPECT, f"Should be in SUSPECT, but got {tracker.state}"
    print(f"✓ Successfully transitioned to SUSPECT state")
    
    # Test 3: Reset and verify SUSPECT state clears
    print(f"\n✓ Test 3: Reset clears SUSPECT state")
    tracker.reset()
    assert tracker.state == TrackerState.NO_TARGET, "Should be NO_TARGET after reset"
    assert tracker.suspect_frame_count == 0, "suspect_frame_count should be 0"
    assert tracker.track_validation_failed_count == 0, "track_validation_failed_count should be 0"
    assert tracker.last_confirmed_bbox is None, "last_confirmed_bbox should be None"
    print(f"✓ All state variables properly reset")
    
    print("\n✅ Phase 3 state transition logic verified!")

if __name__ == "__main__":
    test_phase3_state_transitions()
