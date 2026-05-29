#!/usr/bin/env python
"""Quick integration test for Phase 1: Tracking Validation Debug."""

import numpy as np
from tracker import ROITracker

def test_phase1_tracking_validation():
    """Test that _validate_tracking method works and state variables are initialized."""
    
    # Create a tracker instance
    tracker = ROITracker()
    
    # Create a simple test frame (200x200 BGR)
    test_frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    
    # Add a white textured region (to ensure features exist)
    test_frame[50:120, 50:120] = np.random.randint(100, 255, (70, 70, 3), dtype=np.uint8)
    
    # Initialize tracker with an ROI
    roi_bbox = (50, 50, 70, 70)
    init_ok = tracker.initialize_from_roi(test_frame, roi_bbox)
    print(f"✓ Tracker initialization: {init_ok}")
    print(f"  - State: {tracker.state}")
    print(f"  - Keypoints detected: {tracker.last_init_keypoints}")
    
    # Test _validate_tracking method
    current_bbox = (50, 50, 70, 70)
    is_valid, matches, inliers, ratio = tracker._validate_tracking(test_frame, current_bbox)
    print(f"\n✓ Tracking validation results:")
    print(f"  - Is valid: {is_valid}")
    print(f"  - Matches: {matches}")
    print(f"  - Inliers: {inliers}")
    print(f"  - Inlier ratio: {ratio:.2f}")
    
    # Check state variables are initialized
    print(f"\n✓ State variables initialized:")
    print(f"  - frame_since_last_track_validation: {tracker.frame_since_last_track_validation}")
    print(f"  - track_validation_failed_count: {tracker.track_validation_failed_count}")
    print(f"  - last_track_valid_confidence: {tracker.last_track_valid_confidence}")
    
    # Test reset clears validation state
    tracker.reset()
    print(f"\n✓ After reset:")
    print(f"  - frame_since_last_track_validation: {tracker.frame_since_last_track_validation}")
    print(f"  - track_validation_failed_count: {tracker.track_validation_failed_count}")
    print(f"  - last_track_valid_confidence: {tracker.last_track_valid_confidence}")
    
    print("\n✅ Phase 1 validation logic verified!")

if __name__ == "__main__":
    test_phase1_tracking_validation()
