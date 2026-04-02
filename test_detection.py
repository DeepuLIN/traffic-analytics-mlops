import os
import pytest

# 🚀 Skip BEFORE import
if os.getenv("CI") == "true":
    pytest.skip("Skipping heavy YOLO test in CI", allow_module_level=True)

import numpy as np
from src.detection.yolo_detector import YOLODetector


def test_detection_output():
    detector = YOLODetector()

    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    detections = detector.detect_frame(frame)

    assert detections is not None
    assert isinstance(detections, list)