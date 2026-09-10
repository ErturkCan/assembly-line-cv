"""Render a public sample with pretrained COCO weights, not defect weights.

Run: python demo.py --model /path/to/yolov8n.pt
The sample image ships with Ultralytics. No company footage is used.
"""

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO
from ultralytics.utils import ASSETS

from utils.visualize import draw_detections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--output", type=Path, default=Path("demo"))
    args = parser.parse_args()
    source = ASSETS / "bus.jpg"
    frame = cv2.imread(str(source))
    if frame is None:
        raise RuntimeError(f"Cannot read sample: {source}")
    model = YOLO(args.model)
    result = model(frame, conf=0.4, device="cpu", verbose=False)[0]
    annotated = draw_detections(frame, result)
    args.output.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output / "coco-sample.jpg"), annotated):
        raise RuntimeError("Cannot save sample output")
    records = [{"class": result.names[int(b.cls[0])], "confidence": float(b.conf[0]),
                "xyxy": b.xyxy[0].tolist()} for b in result.boxes]
    metadata = {"model": Path(args.model).name, "source": "Ultralytics assets/bus.jpg",
                "source_url": "https://github.com/ultralytics/ultralytics/blob/main/ultralytics/assets/bus.jpg",
                "scope": "Pretrained COCO inference demo. Not a defect model or a production benchmark.",
                "detections": records}
    (args.output / "result.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved public-sample preview: {len(records)} detections")


if __name__ == "__main__":
    main()
