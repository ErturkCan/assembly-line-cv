import argparse
import json
from ultralytics import YOLO


def evaluate(model_path, data_yaml, img_size, conf_threshold):
    model = YOLO(model_path)

    metrics = model.val(
        data=data_yaml,
        imgsz=img_size,
        conf=conf_threshold,
        plots=True,
    )

    results = {
        "mAP@0.5":    round(metrics.box.map50, 4),
        "mAP@0.5:0.95": round(metrics.box.map, 4),
        "precision":  round(metrics.box.mp, 4),
        "recall":     round(metrics.box.mr, 4),
    }

    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        print(f"  {k}: {v}")

    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to eval_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best.pt")
    parser.add_argument("--data", default="data/defects.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.4)
    args = parser.parse_args()

    evaluate(args.model, args.data, args.imgsz, args.conf)
