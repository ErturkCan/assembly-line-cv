import argparse
import time
import cv2
from ultralytics import YOLO
from utils.visualize import draw_detections
from utils.logger import DetectionLogger

def run(source, model_path, conf_threshold, save_output):
    model = YOLO(model_path)
    logger = DetectionLogger("detections.log")

    # Open video file or webcam (0 = default camera)
    cap = cv2.VideoCapture(source if source != "0" else 0)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    fps_counter = 0
    fps_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf_threshold, verbose=False)[0]

        annotated = draw_detections(frame, results)

        # Calculate and overlay FPS
        fps_counter += 1
        elapsed = time.time() - fps_start
        fps = fps_counter / elapsed if elapsed > 0 else 0
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Log any detections found in this frame
        if len(results.boxes) > 0:
            logger.log(results)

        cv2.imshow("Defect Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished. Average FPS: {fps:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run defect detection on a video source")
    parser.add_argument("--source", required=True, help="Video file path or '0' for webcam")
    parser.add_argument("--model", default="models/best.pt", help="Path to trained YOLO weights")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold (0-1)")
    parser.add_argument("--save", action="store_true", help="Save annotated output to file")
    args = parser.parse_args()

    run(args.source, args.model, args.conf, args.save)
