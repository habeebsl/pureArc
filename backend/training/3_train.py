"""
Train the custom YOLOv8 rim detector.

What this script does, in order:
  1. Converts the Roboflow COCO dataset → YOLO label format
  2. Splits into 80 % train / 20 % val (reproducible random seed)
  3. Writes rim.yaml pointing at the prepared split
  4. Trains YOLOv8-nano from COCO pretrained weights

Trained weights land at:
  backend/runs/rim_detector/weights/best.pt

main.py and debug_rim.py auto-detect that path and switch to the
custom model automatically — no other changes needed.

Usage
-----
  # from backend/
  python training/3_train.py
  python training/3_train.py --epochs 100 --batch 8 --device cpu
"""

import argparse
import json
import os
import random
import shutil

from ultralytics import YOLO

# ── paths ────────────────────────────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
COCO_DIR  = os.path.join(HERE, "hoop_detection_dataset", "train")
COCO_JSON = os.path.join(COCO_DIR, "_annotations.coco.json")
PREPARED  = os.path.join(HERE, "prepared")          # YOLO-format output
YAML_PATH = os.path.join(PREPARED, "rim.yaml")
RUNS      = os.path.join(HERE, "..", "runs")
BASE_WEIGHTS = os.path.join(HERE, "..", "weights", "yolov8n.pt")

SEED      = 42
VAL_SPLIT = 0.20

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--batch",  type=int, default=8)
parser.add_argument("--imgsz",  type=int, default=640)
parser.add_argument("--device", type=str, default="cpu", help="0=GPU  cpu=CPU")
parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
args = parser.parse_args()


# ── Step 1: COCO → YOLO conversion ───────────────────────────────────────────
def convert_coco_to_yolo(coco_json_path: str, images_dir: str, out_dir: str):
    """
    Converts a COCO bbox annotation file to per-image YOLO .txt files.
    Returns list of (image_path, label_path) tuples for all converted images.
    """
    with open(coco_json_path) as f:
        coco = json.load(f)

    # Only keep the 'hoop' category (id=1); ignore the supercategory entry
    hoop_cat_id = next(c["id"] for c in coco["categories"] if c["name"] == "hoop")

    id_to_img = {img["id"]: img for img in coco["images"]}

    # Group annotations by image
    ann_by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        if ann["category_id"] != hoop_cat_id:
            continue
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    os.makedirs(out_dir, exist_ok=True)
    pairs = []

    for img_id, img_info in id_to_img.items():
        img_path = os.path.join(images_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            continue

        W, H      = img_info["width"], img_info["height"]
        anns      = ann_by_image.get(img_id, [])
        stem      = os.path.splitext(img_info["file_name"])[0]
        label_path = os.path.join(out_dir, stem + ".txt")

        lines = []
        for ann in anns:
            x, y, w, h = [float(v) for v in ann["bbox"]]   # COCO: top-left x,y + w,h
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            nw = w / W
            nh = h / H
            # clamp to [0,1] in case of annotation drift at borders
            cx, cy, nw, nh = (max(0., min(1., v)) for v in (cx, cy, nw, nh))
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        with open(label_path, "w") as f:
            f.write("\n".join(lines))

        pairs.append((img_path, label_path))

    return pairs


print("=" * 60)
print("Step 1 — Converting COCO annotations to YOLO format")
print("=" * 60)
raw_labels_dir = os.path.join(PREPARED, "_raw_labels")
pairs = convert_coco_to_yolo(COCO_JSON, COCO_DIR, raw_labels_dir)
print(f"  Converted {len(pairs)} images")


# ── Step 2: 80/20 train/val split ────────────────────────────────────────────
print("\nStep 2 — Splitting into train / val (80/20)")

random.seed(SEED)
random.shuffle(pairs)
split_at   = int(len(pairs) * (1 - VAL_SPLIT))
splits     = {"train": pairs[:split_at], "val": pairs[split_at:]}

for split, items in splits.items():
    img_dir = os.path.join(PREPARED, "images", split)
    lbl_dir = os.path.join(PREPARED, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for src_img, src_lbl in items:
        shutil.copy(src_img, os.path.join(img_dir, os.path.basename(src_img)))
        shutil.copy(src_lbl, os.path.join(lbl_dir, os.path.basename(src_lbl)))

    print(f"  {split:6s}: {len(items)} images")

shutil.rmtree(raw_labels_dir)   # clean up temp label dir


# ── Step 3: write rim.yaml ────────────────────────────────────────────────────
yaml_content = f"""# YOLOv8 dataset — basketball rim detector
path: {PREPARED}
train: images/train
val:   images/val

nc: 1
names:
  0: hoop
"""
with open(YAML_PATH, "w") as f:
    f.write(yaml_content)
print(f"\nWrote {YAML_PATH}")


# ── Step 4: train ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 3 — Training YOLOv8-nano rim detector")
print("=" * 60)

LAST_CKPT = os.path.join(RUNS, "rim_detector", "weights", "last.pt")
if args.resume and os.path.exists(LAST_CKPT):
    print(f"  Resuming from {LAST_CKPT}")
    model = YOLO(LAST_CKPT)
else:
    model = YOLO(BASE_WEIGHTS)   # nano: fast, plenty for a single class

model.train(
    data      = YAML_PATH,
    epochs    = args.epochs,
    imgsz     = args.imgsz,
    batch     = args.batch,
    device    = args.device,
    project   = RUNS,
    name      = "rim_detector",
    exist_ok  = True,
    resume    = args.resume,

    # Augmentation — handles varied gym lighting, camera angles, distances
    hsv_h     = 0.015,
    hsv_s     = 0.4,
    hsv_v     = 0.4,
    fliplr    = 0.5,    # shots from left and right sides
    flipud    = 0.0,    # rim is always above floor
    mosaic    = 0.5,
    scale     = 0.4,    # rim appears smaller/larger at different distances
    translate = 0.1,
    degrees   = 5.0,

    cls       = 0.5,
    box       = 7.5,
    patience  = 20,
    save      = True,
    plots     = True,
)

best = os.path.join(RUNS, "rim_detector", "weights", "best.pt")
print(f"\nDone!  Best weights → {best}")
print("main.py and debug_rim.py will pick them up automatically.")
