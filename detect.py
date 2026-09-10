import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO
from utils.visualize import draw_detections
from utils.logger import DetectionLogger


def run(source, model_path, conf_threshold=0.4, save_output=None, display=True):
    if save_output and str(source) != "0" and Path(source).resolve() == Path(save_output).resolve():
        raise ValueError("Output video must be different from the input video")
    model = YOLO(model_path)
    logger = DetectionLogger("detections.log")
    cap = cv2.VideoCapture(0 if str(source) == "0" else source)
    writer = None
    frame_count = 0
    start = time.perf_counter()
    fps = 0.0
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, conf=conf_threshold, verbose=False)[0]
            annotated = draw_detections(frame, results)
            frame_count += 1
            elapsed = time.perf_counter() - start
            fps = frame_count / elapsed if elapsed > 0 else 0.0
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if len(results.boxes) > 0:
                logger.log(results)
            if save_output:
                if writer is None:
                    height, width = annotated.shape[:2]
                    source_fps = cap.get(cv2.CAP_PROP_FPS)
                    if not 0 < source_fps < float("inf"):
                        source_fps = 30.0
                    writer = cv2.VideoWriter(str(save_output), cv2.VideoWriter_fourcc(*"mp4v"),
                                             source_fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError(f"Cannot write video: {save_output}")
                writer.write(annotated)
            if display:
                cv2.imshow("Defect Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()
    print(f"Finished. Frames: {frame_count}; average loop FPS: {fps:.1f}")
    return frame_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run defect detection on a video source")
    parser.add_argument("--source", required=True, help="Video file path or '0' for webcam")
    parser.add_argument("--model", required=True, help="Path to trained YOLO weights")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold (0-1)")
    parser.add_argument("--save", metavar="PATH", nargs="?", const="annotated.mp4", help="Save an annotated MP4")
    parser.add_argument("--no-display", action="store_true", help="Run without opening a preview window")
    args = parser.parse_args()
    run(args.source, args.model, args.conf, args.save, display=not args.no_display)
