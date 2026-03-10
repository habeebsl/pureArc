"""
Train a YOLOv8 shot detector — Basketball + Basketball Hoop.

This produces a single model that detects both objects, replacing the
separate YOLOWorld ball detector and enabling Avi Shah-style trajectory
scoring as an alternative / supplement to net_motion.py.

Trained weights land at:
  backend/runs/shot_detector/weights/best.pt

Usage
-----
  # from backend/
  python training/4_train_shot.py
  python training/4_train_shot.py --epochs 100 --batch 8 --device cpu
"""

import argparse
import os

from ultralytics import YOLO

HERE      = os.path.dirname(os.path.abspath(__file__))
YAML_PATH = os.path.join(HERE, "dataset", "shot_config_1300.yaml")
RUNS_DIR  = os.path.join(HERE, "..", "runs")
_EXISTING_WEIGHTS = os.path.join(HERE, "..", "runs", "shot_detector", "weights", "best.pt")
_DEFAULT_BASE = os.path.join(HERE, "..", "weights", "yolov8n.pt")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch",  type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--base",   type=str,
                        default=_EXISTING_WEIGHTS if os.path.exists(_EXISTING_WEIGHTS) else _DEFAULT_BASE,
                        help="Weights to fine-tune from (defaults to existing best.pt)")
    args = parser.parse_args()

    print(f"Config : {YAML_PATH}")
    print(f"Base   : {args.base}")
    print(f"Epochs : {args.epochs}  Batch: {args.batch}  Device: {args.device}")

    model = YOLO(args.base)
    model.train(
        data=YAML_PATH,
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        device=args.device,
        project=RUNS_DIR,
        name="shot_detector",
        exist_ok=True,
    )
    print(f"\nDone. Best weights → {RUNS_DIR}/shot_detector/weights/best.pt")

if __name__ == "__main__":
    main()
