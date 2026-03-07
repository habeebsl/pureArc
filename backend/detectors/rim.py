"""
Rim (basketball hoop ring) detector — v2.

Design rationale (informed by Mi Research BSD paper, §3.1)
----------------------------------------------------------
The paper trains a *custom* YOLO specifically for hoops so its bbox wraps
the ring tightly by construction.  We are using YOLOWorld (open-vocabulary),
which was never trained for hoops — it boxes the whole assembly:
backboard + rim + net, placing the centroid in the middle of the backboard.

Fix: two-stage pipeline
  Stage 1 — YOLO coarse ROI
    YOLOWorld finds the whole hoop assembly.  We use its bbox ONLY as a
    region-of-interest and intentionally discard its centroid / bbox for
    distance calculations.

  Stage 2 — Orange-ring localisation
    Inside the LOWER HALF of the YOLO ROI (the rim ring is never at the
    top of a "basketball hoop" detection — that's the backboard), we run:
      a. HSV mask for the orange steel rim colour.
      b. Morphological clean-up.
      c. Contour filter: aspect ratio 1.3–7 (ring is wide, not tall) and
         minimum area gate to reject noise.
      d. Best candidate = highest fill-ratio (most ring-like contour).
      e. Return bbox tightly hugging that contour (+small padding).
    Falls back to full-frame search if YOLO finds nothing.

  Stage 3 — Object tracking lock
    After _LOCK_THRESH consecutive stable YOLO detections we hand off to an
    OpenCV CSRT tracker.  CSRT follows the rim's visual appearance across
    every frame so the bbox moves correctly with camera movement.
    Every _REDETECT_EVERY frames YOLO re-runs to correct tracker drift.
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO, YOLOWorld


# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------
_YOLO_CLASSES   = ["basketball hoop", "basketball rim"]
_YOLO_CONF      = 0.10          # intentionally low — rim can be far/small
_CUSTOM_CONF    = 0.10          # raised from 0.05 — reduces false positives while
                                # still capturing detections in the 0.11-0.17 conf range
_LOCK_THRESH     = 5            # consecutive stable YOLO detections before handing to tracker
_LOCK_RADIUS_PX  = 40           # max centroid drift (px) to still count as stable
_MAX_TRACK_FAILS = 8            # consecutive tracker-only failures before full reset
_YOLO_SNAP_RADIUS = 120         # px — YOLO must be within this of tracker to snap to it

# HSV range for the orange steel rim
_HSV_LOW  = np.array([ 5,  90,  60], dtype=np.uint8)
_HSV_HIGH = np.array([28, 255, 255], dtype=np.uint8)

# Aspect ratio gate: rim ring viewed in perspective is always wider than tall
_ASPECT_MIN = 1.3
_ASPECT_MAX = 7.0

# Contour area gate (px²) at 640×480
#   Real rim ring:  ~600 – 8000 px²  (varies with distance / zoom)
#   Floor splash / jersey: typically > 10 000 px²
#   Emoji / tiny decoration: < 200 px²
_MIN_AREA = 200
_MAX_AREA = 9000

# Rim must be at least this wide in pixels — filters out small emoji/icons
_MIN_WIDTH_PX = 20

# Rim is never in the bottom fraction of the frame (that's the floor)
_MAX_Y_FRAC = 0.82    # centre y must be above this fraction of frame height

# Fraction of YOLO ROI height to skip from the top (backboard region)
_SKIP_TOP_FRAC = 0.45


_WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")


class RimDetector:
    """
    Detects the basketball rim *ring* and returns its tight pixel bbox.

    Pass custom_model_path to use a trained YOLOv8 rim detector (recommended).
    Falls back to the two-stage YOLOWorld + HSV pipeline when not provided.
    """

    def __init__(self, model_path: str = "yolov8x-worldv2",
                 custom_model_path: str = None):
        self._custom = custom_model_path is not None
        if self._custom:
            self._model = YOLO(custom_model_path)
            print(f"RimDetector: using custom trained model → {custom_model_path}")
        else:
            self._model = YOLOWorld(os.path.join(_WEIGHTS_DIR, model_path + ".pt"))
            self._model.set_classes(_YOLO_CLASSES)
            print("RimDetector: using YOLOWorld + HSV fallback pipeline")

        self._locked           = False
        self._tracker          = None   # OpenCV CSRT tracker (active when locked)
        self._tracked_bbox     = None   # current tracker bbox (x1,y1,x2,y2)
        self._stable_count     = 0
        self._frame_since_lock = 0
        self._prev_center      = None
        self._track_fail_count = 0      # consecutive tracker update failures

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_rim(self, frame: np.ndarray) -> "dict | None":
        """
        Parameters
        ----------
        frame : BGR numpy array

        Returns
        -------
        dict  {"center": (cx, cy), "bbox": (x1,y1,x2,y2), "locked": bool}
        or None if the rim ring is not found.
        """
        if self._locked:
            self._frame_since_lock += 1

            # --- YOLO-primary: try a fresh detection every frame ---
            fresh = self._full_detect(frame)

            # Advance tracker regardless (keeps it warm for gap-filling)
            ok, rect = self._tracker.update(frame)
            if ok:
                rx, ry, rw, rh = [int(v) for v in rect]
                tracker_bbox = (rx, ry, rx + rw, ry + rh)
                tracker_cx   = rx + rw // 2
                tracker_cy   = ry + rh // 2
            else:
                tracker_bbox = self._tracked_bbox
                tracker_cx, tracker_cy = ((self._tracked_bbox[0] + self._tracked_bbox[2]) // 2,
                                          (self._tracked_bbox[1] + self._tracked_bbox[3]) // 2)

            if fresh is not None:
                fx, fy = fresh["center"]
                # Sanity-check: YOLO result must be near where tracker thinks rim is
                if np.hypot(fx - tracker_cx, fy - tracker_cy) < _YOLO_SNAP_RADIUS:
                    # YOLO wins — its bbox is always more precisely centred on the rim
                    self._init_tracker(frame, fresh["bbox"])
                    self._tracked_bbox = fresh["bbox"]
                    self._track_fail_count = 0
                    return {"center": fresh["center"], "bbox": fresh["bbox"], "locked": True}

            # YOLO missed this frame — fall back to tracker
            if not ok:
                self._track_fail_count += 1
                if self._track_fail_count >= _MAX_TRACK_FAILS:
                    self.reset_lock()
                    return None
            else:
                self._track_fail_count = 0
                self._tracked_bbox = tracker_bbox

            x1, y1, x2, y2 = self._tracked_bbox
            return {"center": ((x1+x2)//2, (y1+y2)//2),
                    "bbox":   self._tracked_bbox,
                    "locked": True}

        # --- Not locked yet: run YOLO and wait for _LOCK_THRESH stable hits ---
        result = self._full_detect(frame)
        if result is None:
            self._stable_count = 0
            self._prev_center  = None
            return None

        cx, cy = result["center"]
        if self._prev_center is not None:
            drift = np.hypot(cx - self._prev_center[0], cy - self._prev_center[1])
            self._stable_count = self._stable_count + 1 if drift < _LOCK_RADIUS_PX else 0
        self._prev_center = (cx, cy)

        if self._stable_count >= _LOCK_THRESH:
            self._locked           = True
            self._tracked_bbox     = result["bbox"]
            self._frame_since_lock = 0
            self._track_fail_count = 0
            self._init_tracker(frame, result["bbox"])
            result["locked"] = True
        else:
            result["locked"] = False

        return result

    def reset_lock(self):
        """Force the detector to re-search (e.g. after a camera cut)."""
        self._locked           = False
        self._tracker          = None
        self._tracked_bbox     = None
        self._stable_count     = 0
        self._prev_center      = None
        self._track_fail_count = 0

    # ------------------------------------------------------------------
    def _init_tracker(self, frame: np.ndarray, bbox: tuple):
        """Initialise (or re-initialise) the MIL tracker on *bbox*."""
        x1, y1, x2, y2 = bbox
        self._tracker = cv2.TrackerMIL_create()
        self._tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _full_detect(self, frame: np.ndarray) -> "dict | None":
        """
        Custom model: direct tight detection, no HSV needed.
        Fallback pipeline: YOLO coarse ROI → orange-ring HSV localisation.
        """
        if self._custom:
            return self._run_custom(frame)

        H, W = frame.shape[:2]
        yolo_bbox = self._yolo_roi(frame)   # (x1,y1,x2,y2) or None

        if yolo_bbox is not None:
            rx1, ry1, rx2, ry2 = yolo_bbox
            # Skip the top portion — that's backboard, not rim
            skip = int((ry2 - ry1) * _SKIP_TOP_FRAC)
            search_y1 = ry1 + skip
            crop   = frame[search_y1:ry2, rx1:rx2]
            offset = (rx1, search_y1)
            result = self._orange_ring(crop, offset)
            if result is not None:
                return result
            # YOLO ROI was too tight — retry on full frame

        return self._orange_ring(frame, (0, 0))

    def _run_custom(self, frame: np.ndarray) -> "dict | None":
        """Direct inference with the custom-trained rim detector."""
        results = self._model(frame, conf=_CUSTOM_CONF, verbose=False)[0]
        best_area = -1
        best      = None
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best = {
                    "center": (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                    "bbox":   (int(x1), int(y1), int(x2), int(y2)),
                    "locked": False,
                }
        return best

    def _yolo_roi(self, frame: np.ndarray) -> "tuple | None":
        """Returns the raw YOLO bbox as ROI only — never used directly as rim bbox."""
        results = self._model(frame, conf=_YOLO_CONF, verbose=False)[0]
        best_area = -1
        best_bbox = None
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_bbox = (int(x1), int(y1), int(x2), int(y2))
        return best_bbox

    def _orange_ring(self, crop: np.ndarray, offset: tuple) -> "dict | None":
        """
        HSV mask → morphology → contour filter → tight bbox around the rim ring.
        *offset* = (x_off, y_off) translates crop-local coords back to frame space.
        """
        if crop.size == 0:
            return None

        ox, oy = offset
        H_c, W_c = crop.shape[:2]

        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, _HSV_LOW, _HSV_HIGH)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        H_frame = oy + H_c   # absolute bottom of search region in frame space
        max_cy_px = int(_MAX_Y_FRAC * (oy + H_c)) if oy == 0 else H_frame

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (_MIN_AREA <= area <= _MAX_AREA):
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if h == 0 or w < _MIN_WIDTH_PX:
                continue
            aspect = w / h
            if not (_ASPECT_MIN <= aspect <= _ASPECT_MAX):
                continue
            # Reject detections in the bottom portion of the full frame
            abs_cy = oy + y + h // 2
            if oy == 0 and abs_cy > max_cy_px:   # only apply when searching full frame
                continue
            # Fill ratio: how much of the bounding rect is actually orange
            fill = area / (w * h)
            candidates.append((fill, area, cnt))

        if not candidates:
            return None

        # Best = highest fill ratio (most compact ring-like shape), ties broken by area
        candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
        _, _, best_cnt = candidates[0]

        x, y, w, h = cv2.boundingRect(best_cnt)

        # Small uniform padding so the box sits just outside the orange ring
        pad = max(4, int(h * 0.3))
        x1 = max(0,   ox + x - pad)
        y1 = max(0,   oy + y - pad)
        x2 = min(ox + W_c, ox + x + w + pad)
        y2 = min(oy + H_c, oy + y + h + pad)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        return {"center": (cx, cy), "bbox": (x1, y1, x2, y2), "locked": False}
