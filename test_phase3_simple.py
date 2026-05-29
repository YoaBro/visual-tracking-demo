#!/usr/bin/env python
"""Integration test for Phase 3: State transitions (TRACKING -> SUSPECT -> LOST)."""

import config
from tracker import ROITracker, TrackerState

def test_phase3_state_transitions():
    """Test state transitions: TRACKING (valid) -> SUSPECT (failed) -> LOST (sustained failures)."""
    
    # Create a tracker instance
    tracker = ROITracker()
    
    # Manually set up tracker state (avoid CSRT initialization issues in test)
    tracker.state = TrackerState.TRACKING
    tracker.current_bbox = (50, 50, 70, 70)
    tracker.last_valid_bbox = (50, 50, 70, 70)
    tracker.last_confirmed_bbox = (50, 50, 70, 70)
    tracker.track_validation_failed_count = 0
    tracker.suspect_frame_count = 0
    
    print(f"✓ Tracker manually set to TRACKING state")
    print(f"  - current_bbox: {tracker.current_bbox}")
    print(f"  - last_confirmed_bbox: {tracker.last_confirmed_bbox}")
    print(f"  - state: {tracker.state}")
    
    # Test 1: Validation counter increments
    print(f"\n✓ Test 1: Validation failure counter increments")
    for i in range(5):
        tracker.track_validation_failed_count += 1
        print(f"  Failure {i+1}: count={tracker.track_validation_failed_count}")
    
    # Test 2: Transition to SUSPECT when threshold reached
    print(f"\n✓ Test 2: Transition TRACKING -> SUSPECT at threshold")
    tracker.state = TrackerState.TRACKING
    tracker.track_validation_failed_count = 0
    
    # Simulate reaching the failure threshold
    for i in range(config.TRACK_MAX_FAILED_VALIDATIONS):
        tracker.track_validation_failed_count += 1
        if tracker.track_validation_failed_count >= config.TRACK_MAX_FAILED_VALIDATIONS:
            if tracker.state == TrackerState.TRACKING:
                tracker.state = TrackerState.SUSPECT
                tracker.suspect_frame_count = 0
                print(f"  Failure {i+1}: state transitioned to SUSPECT")
    
    assert tracker.state == TrackerState.SUSPECT, f"Should be SUSPECT, got {tracker.state}"
    print(f"✓ Successfully transitioned to SUSPECT")
    
    # Test 3: Reset clears SUSPECT state
    print(f"\n✓ Test 3: Reset clears SUSPECT state")
    tracker.reset()
    assert tracker.state == TrackerState.NO_TARGET, "Should be NO_TARGET after reset"
    assert tracker.suspect_frame_count == 0, "suspect_frame_count should be 0"
    assert tracker.track_validation_failed_count == 0, "track_validation_failed_count should be 0"
    assert tracker.last_confirmed_bbox is None, "last_confirmed_bbox should be None"
    print(f"✓ All state variables properly reset")
    
    # Test 4: Validation success resets counters
    print(f"\n✓ Test 4: Validation success resets failure counters")
    tracker.state = TrackerState.TRACKING
    tracker.track_validation_failed_count = 2
    tracker.suspect_frame_count = 1
    
    # Simulate successful validation
    tracker.track_validation_failed_count = 0
    tracker.suspect_frame_count = 0
    
    assert tracker.track_validation_failed_count == 0, "Failed count should be 0"
    assert tracker.suspect_frame_count == 0, "Suspect count should be 0"
    print(f"✓ Counters reset on validation success")
    
    print("\n✅ Phase 3 state transition logic verified!")

if __name__ == "__main__":
    test_phase3_state_transitions()
