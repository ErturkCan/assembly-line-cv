import cv2
import numpy as np

# Color per defect class (BGR)
CLASS_COLORS = {
    "scratch":       (0, 0, 255),
    "dent":          (0, 165, 255),
    "paint_bubble":  (0, 255, 255),
    "contamination": (255, 0, 0),
}
DEFAULT_COLOR = (0, 255, 0)


def draw_detections(frame, results):
    annotated = frame.copy()

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        label = results.names[class_id]

        color = CLASS_COLORS.get(label, DEFAULT_COLOR)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(annotated, text, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return annotated
