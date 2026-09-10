import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from detect import run


class VideoTests(unittest.TestCase):
    def test_empty_video_releases_capture(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (False, None)
        with patch("detect.YOLO"), patch("detect.DetectionLogger"), patch("detect.cv2.VideoCapture", return_value=capture):
            self.assertEqual(run("input.mp4", "weights.pt", display=False), 0)
        capture.release.assert_called_once()

    def test_headless_video_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            source, target = Path(d) / "in.mp4", Path(d) / "out.mp4"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
            self.assertTrue(writer.isOpened())
            for _ in range(3):
                writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
            writer.release()
            model = Mock(return_value=[SimpleNamespace(boxes=[])])
            with patch("detect.YOLO", return_value=model), patch("detect.DetectionLogger"):
                self.assertEqual(run(str(source), "weights.pt", save_output=str(target), display=False), 3)
            capture = cv2.VideoCapture(str(target))
            count = 0
            while capture.read()[0]:
                count += 1
            capture.release()
            self.assertEqual(count, 3)

    def test_cannot_overwrite_input(self):
        with self.assertRaises(ValueError):
            run("same.mp4", "weights.pt", save_output="same.mp4", display=False)
