"""
Step 1 — Extract frames from all videos for labelling.

Saves to  training/dataset/images/raw/
Every Nth frame is sampled (default N=5) to avoid near-duplicate frames
while capturing the full range of rim positions and lighting conditions.

Usage
-----
  python training/1_extract_frames.py           # default N=5, all videos
  python training/1_extract_frames.py 8         # sample every 8th frame
"""

import cv2
import os
import sys

VIDEOS = [
    "test_shot.mp4",
    "test_shot2.mp4",
    "test_shot3.mp4",
]

SAMPLE_EVERY = int(sys.argv[1]) if len(sys.argv) > 1 else 5
OUT_DIR = os.path.join(os.path.dirname(__file__), "dataset", "images", "raw")
os.makedirs(OUT_DIR, exist_ok=True)

total_saved = 0

for video_path in VIDEOS:
    full_path = os.path.join(os.path.dirname(__file__), "..", video_path)
    if not os.path.exists(full_path):
        print(f"  [skip] {video_path} not found")
        continue

    cap = cv2.VideoCapture(full_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name   = os.path.splitext(video_path)[0]
    saved        = 0

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i % SAMPLE_EVERY != 0:
            continue
        frame = cv2.resize(frame, (640, 480))
        fname = os.path.join(OUT_DIR, f"{video_name}_f{i:04d}.jpg")
        cv2.imwrite(fname, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        saved += 1

    cap.release()
    print(f"  {video_path}: {total_frames} frames → {saved} saved")
    total_saved += saved

print(f"\nDone — {total_saved} images in {OUT_DIR}")
print(f"\nNext step: label these images with labelImg")
print(f"  pip install labelImg")
print(f"  labelImg {OUT_DIR} {os.path.join(os.path.dirname(__file__), 'classes.txt')}")
