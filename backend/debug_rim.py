"""
Quick rim-detection debugger — no waiting for a full video.

Samples N frames spread evenly across the video, runs RimDetector on each,
and saves annotated snapshots to rim_debug/.

Usage
-----
  python debug_rim.py                    # uses test_shot3.mp4, 5 frames
  python debug_rim.py test_shot2.mp4 8  # custom video, 8 frames
"""

import sys
import os
import cv2
import numpy as np
from rim import RimDetector, _HSV_LOW, _HSV_HIGH

_CUSTOM_WEIGHTS = os.path.join(os.path.dirname(__file__), "runs", "rim_detector", "weights", "best.pt")


VIDEO   = sys.argv[1] if len(sys.argv) > 1 else "test_shot4.mp4"
N_SNAPS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
OUT_DIR = "rim_debug"
os.makedirs(OUT_DIR, exist_ok=True)

# Clear any previous debug images
for f in os.listdir(OUT_DIR):
    os.remove(os.path.join(OUT_DIR, f))
print(f"Cleared {OUT_DIR}/")

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    print(f"ERROR: cannot open {VIDEO}")
    sys.exit(1)

total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
sample_frames = [int(total * i / N_SNAPS) for i in range(N_SNAPS)]
print(f"Video: {VIDEO}  |  total frames: {total}  |  sampling: {sample_frames}")

detector = RimDetector(
    custom_model_path=_CUSTOM_WEIGHTS if os.path.exists(_CUSTOM_WEIGHTS) else None
)

for frame_idx in sample_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        print(f"  [frame {frame_idx}] could not read — skipping")
        continue

    frame = cv2.resize(frame, (640, 480))
    result = detector.detect_rim(frame)

    # ------------------------------------------------------------------ #
    # Annotated output frame
    # ------------------------------------------------------------------ #
    vis = frame.copy()

    if result:
        x1, y1, x2, y2 = result["bbox"]
        cx, cy          = result["center"]
        locked_str      = "LOCKED" if result["locked"] else "searching"

        # Tight bbox around the ring in bright green
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Centre crosshair
        cv2.drawMarker(vis, (cx, cy), (0, 255, 0),
                       cv2.MARKER_CROSS, 14, 2)
        cv2.putText(vis, f"RIM [{locked_str}]", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        print(f"  [frame {frame_idx:04d}] rim @ ({cx},{cy})  bbox=({x1},{y1},{x2},{y2})  {locked_str}")
    else:
        cv2.putText(vis, "RIM: NOT FOUND", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        print(f"  [frame {frame_idx:04d}] NOT FOUND")

    out_path = os.path.join(OUT_DIR, f"debug_frame_{frame_idx:04d}.png")
    cv2.imwrite(out_path, vis)
    print(f"             → saved {out_path}")

    # ------------------------------------------------------------------ #
    # HSV mask diagnostic (helps tune colour range)
    # ------------------------------------------------------------------ #
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _HSV_LOW, _HSV_HIGH)
    mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # Overlay the orange pixels in red on a dark copy of the frame
    dark = (frame * 0.3).astype(np.uint8)
    dark[mask > 0] = (0, 80, 255)   # bright red-orange for orange pixels
    if result:
        x1, y1, x2, y2 = result["bbox"]
        cv2.rectangle(dark, (x1, y1), (x2, y2), (0, 255, 0), 2)
    diag_path = os.path.join(OUT_DIR, f"debug_hsv_{frame_idx:04d}.png")
    cv2.imwrite(diag_path, dark)
    print(f"             → hsv diag {diag_path}")

cap.release()
print(f"\nDone — {N_SNAPS} snapshots saved to {OUT_DIR}/")
