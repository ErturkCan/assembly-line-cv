import csv
import time
from pathlib import Path


class DetectionLogger:
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        # Write header if file doesn't exist yet
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "class", "confidence", "x1", "y1", "x2", "y2"])

    def log(self, results):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = round(float(box.conf[0]), 4)
                cls = results.names[int(box.cls[0])]
                writer.writerow([timestamp, cls, conf, x1, y1, x2, y2])
