import argparse
from pathlib import Path
from ultralytics import YOLO


def train(data_yaml, model_name, epochs, img_size, batch_size, output_dir):
    model = YOLO(model_name)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project=output_dir,
        name="defect_detector",
        # Save best weights based on validation mAP
        save=True,
        val=True,
        plots=True,
    )

    print(f"\nTraining complete.")
    print(f"Best weights saved to: {Path(output_dir) / 'defect_detector' / 'weights' / 'best.pt'}")
    print(f"mAP@0.5: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 defect detection model")
    parser.add_argument("--data", default="data/defects.yaml", help="Dataset config YAML")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model (n/s/m/l/x)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--output", default="runs/train")
    args = parser.parse_args()

    train(args.data, args.model, args.epochs, args.imgsz, args.batch, args.output)
