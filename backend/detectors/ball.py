import os
import numpy as np
from ultralytics import YOLOWorld

_WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")


class BallDetector:
    def __init__(self, model_size="yolov8m-worldv2", conf=0.15):
        self.model = YOLOWorld(os.path.join(_WEIGHTS_DIR, model_size + ".pt"))
        self.model.set_classes(["basketball"])  # open-vocab: look ONLY for a basketball
        self.conf = conf

        # Temporal tracker
        self._last_cx    = None
        self._last_cy    = None
        self._lost_count = 0
        self._max_lost   = 12
        self._max_jump   = 250  # px — ball moves fast during a shot

        # Minimum radius in pixels to accept a detection as a real basketball.
        # A 6px radius blob is noise; a real ball at typical shooting distance
        # will be at least 10px radius at 640x480.
        self._min_radius = 10

    def detect_ball(self, frame):
        """
        Returns (cx, cy, radius) of the detected basketball, or None.
        """
        results = self.model(frame, conf=self.conf, verbose=False)[0]

        best      = None
        best_area = -1

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            r  = int(max(x2 - x1, y2 - y1) / 2)  # use bounding box to approximate radius
            area = (x2 - x1) * (y2 - y1)

            # Reject tiny blobs — not a basketball
            if r < self._min_radius:
                continue

            # Temporal gate — reject detections that jump too far
            if self._last_cx is not None and self._lost_count < self._max_lost:
                dist = np.hypot(cx - self._last_cx, cy - self._last_cy)
                if dist > self._max_jump:
                    continue

            # Prefer the largest detection (closest / most visible ball)
            if area > best_area:
                best_area = area
                best = (cx, cy, r)

        if best is not None:
            self._last_cx    = best[0]
            self._last_cy    = best[1]
            self._lost_count = 0
        else:
            self._lost_count += 1
            if self._lost_count > self._max_lost:
                self._last_cx = None
                self._last_cy = None

        return best