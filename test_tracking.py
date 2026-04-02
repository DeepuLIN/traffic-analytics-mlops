import os
import pytest

# 🚀 Skip BEFORE heavy imports
if os.getenv("CI") == "true":
    pytest.skip("Skipping DeepSORT test in CI", allow_module_level=True)

import numpy as np
from src.tracking.deepsort_tracker import Tracker
from src.analytics.counter import Analytics


def test_tracking_pipeline():
    tracker = Tracker()
    analytics = Analytics(line_position=300)

    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    dummy_detections = []

    tracks = tracker.update(dummy_detections, dummy_frame)
    stats = analytics.update(tracks)

    assert tracks is not None
    assert stats is not None   # 🔥 you missed this earlier