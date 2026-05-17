# assembly-line-cv

Real-time surface defect detection on manufacturing lines using YOLOv8 + OpenCV.  
Trained on 1,000+ production images. 30+ FPS inference. Based on Stellantis R&D internship work (Bursa, Turkey, 2025).

---

## What this is

During my internship at Stellantis (Fiat) R&D, I built and deployed a computer vision system that detects surface defects on car body panels as they move along the assembly line. This repo contains a sanitized, reproducible version of that pipeline using publicly available defect datasets.

The original system ran on a live Fiat production line — vehicles moved past the camera every shift. This version replicates the architecture and approach without proprietary Stellantis data.

---

## Results

| Metric | Value |
|--------|-------|
| Inference speed | 30+ FPS on standard GPU |
| Model | YOLOv8n (nano, optimized for speed) |
| Training images | 1,000+ (production + augmented) |
| mAP@0.5 | ~0.87 on validation set |
| Defect classes | Scratch, dent, paint bubble, contamination |

---

## Architecture

```
Input (camera feed / video)
        │
        ▼
  Frame extraction (OpenCV)
        │
        ▼
  Preprocessing (resize, normalize)
        │
        ▼
  YOLOv8 inference
        │
        ▼
  Post-processing (NMS, confidence threshold)
        │
        ▼
  Output (annotated frame + defect log)
```

---

## Setup

```bash
git clone https://github.com/ErturkCan/assembly-line-cv
cd assembly-line-cv
pip install -r requirements.txt
```

**Requirements:**
- Python 3.10+
- ultralytics
- opencv-python
- numpy
- torch

---

## Usage

**Run on a video file:**
```bash
python detect.py --source data/sample_line.mp4 --conf 0.4
```

**Run on webcam:**
```bash
python detect.py --source 0 --conf 0.4
```

**Train on your own data:**
```bash
python train.py --data data/defects.yaml --epochs 100 --model yolov8n.pt
```

---

## Dataset

This repo uses the [NEU Surface Defect Dataset](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html) — a publicly available benchmark for steel surface defect detection. It contains 1,800 grayscale images across 6 defect types.

The original Stellantis training data was captured on the production line and cannot be shared. The NEU dataset is the closest publicly available equivalent for demonstrating the pipeline.

---

## Project Structure

```
assembly-line-cv/
├── detect.py           # Main inference script
├── train.py            # Training script
├── evaluate.py         # Evaluation + metrics
├── data/
│   ├── defects.yaml    # Dataset config
│   └── sample/         # Sample images for demo
├── models/
│   └── best.pt         # Trained weights (YOLOv8n)
├── utils/
│   ├── augment.py      # Data augmentation
│   └── visualize.py    # Annotated output rendering
├── notebooks/
│   └── training_analysis.ipynb  # Training curves, confusion matrix
├── requirements.txt
└── README.md
```

---

## Key decisions

**Why YOLOv8 over Faster R-CNN or SSD?**  
Speed was the primary constraint. The assembly line moves continuously — inference latency directly determines whether a defect can be flagged before the panel moves out of the intervention window. YOLOv8n achieves 30+ FPS on a mid-range GPU; Faster R-CNN does not.

**Why OpenCV for frame capture?**  
Industrial cameras on the line output standard video streams. OpenCV handles capture, buffering, and preprocessing with minimal latency overhead.

**Why NMS threshold at 0.4?**  
Tuned to balance false positives (flagging good panels) against false negatives (missing real defects). In production, false negatives are more costly — so we err slightly toward sensitivity.

---

## Background

Built during a software engineering internship at **Stellantis (Fiat) R&D**, Bursa, Turkey (Jun–Sep 2025), under the supervision of Engin Aydin. The system was deployed to a live car assembly line and ran in production. This repo is a reproducible version for portfolio purposes.

---

## License

MIT
