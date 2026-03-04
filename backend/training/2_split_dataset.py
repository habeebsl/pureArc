"""
Step 2 — Split labelled images into train / val sets (80/20).

Run this AFTER you have finished labelling images in labelImg.
labelImg saves a .txt file next to each image in images/raw/.
This script copies images + labels that have a matching .txt file
into images/train|val and labels/train|val.

Usage
-----
  python training/2_split_dataset.py
"""

import os
import shutil
import random

BASE    = os.path.join(os.path.dirname(__file__), "dataset")
RAW_IMG = os.path.join(BASE, "images", "raw")
VAL_PCT = 0.20
SEED    = 42

random.seed(SEED)

# Collect all images that have a matching label file
labelled = []
for fname in sorted(os.listdir(RAW_IMG)):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    stem  = os.path.splitext(fname)[0]
    label = os.path.join(RAW_IMG, stem + ".txt")
    if os.path.exists(label):
        labelled.append(stem)
    else:
        print(f"  [no label] skipping {fname}")

print(f"\nLabelled images found: {len(labelled)}")
if len(labelled) == 0:
    print("Nothing to split — label your images first with labelImg.")
    raise SystemExit(1)

random.shuffle(labelled)
split_at = int(len(labelled) * (1 - VAL_PCT))
splits   = {"train": labelled[:split_at], "val": labelled[split_at:]}

for split, stems in splits.items():
    img_dir = os.path.join(BASE, "images", split)
    lbl_dir = os.path.join(BASE, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for stem in stems:
        # Find the image extension
        for ext in (".jpg", ".jpeg", ".png"):
            src_img = os.path.join(RAW_IMG, stem + ext)
            if os.path.exists(src_img):
                shutil.copy(src_img, os.path.join(img_dir, stem + ext))
                break
        shutil.copy(
            os.path.join(RAW_IMG, stem + ".txt"),
            os.path.join(lbl_dir, stem + ".txt"),
        )

    print(f"  {split:6s}: {len(stems)} images → {img_dir}")

print("\nDone — ready to train.")
print("Next step:  python training/3_train.py")
