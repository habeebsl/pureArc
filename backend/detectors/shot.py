"""
Trajectory-based shot detection using a trained YOLOv8 model that detects
'Basketball' (class 0) and 'Basketball Hoop' (class 1).

Scoring logic — direct rim-crossing detector (camera-angle agnostic):
  WATCHING → ARMED : ball detected above the rim within horizontal hoop bounds
  ARMED    → SCORED: ball crosses downward through the rim plane while aligned
  ARMED    → MISS  : ball exits the hoop area without crossing (armed_timeout)

This replaces the Avi Shah two-phase approach which requires a specific
camera angle.  The crossing check works regardless of camera distance or angle.

Weights: backend/runs/shot_detector/weights/best.pt
Train:   python training/4_train_shot.py
"""

import os

import numpy as np
from ultralytics import YOLO

from .shot_utils import clean_ball_pos, clean_hoop_pos, in_hoop_region

_TRAINED_WEIGHTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "runs", "shot_detector", "weights", "best.pt",
)

_CLS_BALL = 0
_CLS_HOOP = 1

# Horizontal reach beyond the hoop bbox edges that still counts as "in bounds"
_HOOP_H_MARGIN = 0.50   # fraction of hoop width

# How many frames to stay ARMED before giving up and counting a miss
_ARMED_TIMEOUT = 75

# Ball must have moved at least this many pixels across recent detections to
# be considered "in play" — prevents stationary false-positives (e.g. the
# backboard target square) from triggering the ARMED state.
_MIN_BALL_MOTION_PX = 5

# Case B (underhand/from-below shot): if the ball was last detected inside
# the hoop bbox and then disappears for this many frames, count as a make.
_BBOX_VANISH_FRAMES = 10


class ShotDetector:
    """
    Per-frame shot detector.  Call update(frame) every frame.

    State machine:
      WATCHING → ARMED  : ball rises above the rim inside horizontal bounds
      ARMED    → COOLDOWN (make): ball crosses rim plane downward in bounds
      ARMED    → WATCHING (miss): armed_timeout expires without a crossing
    """

    def __init__(self, ball_conf: float = 0.25, hoop_conf: float = 0.30,
                 cooldown_frames: int = 60):
        if not os.path.exists(_TRAINED_WEIGHTS):
            raise FileNotFoundError(
                f"Shot detector weights not found at {_TRAINED_WEIGHTS}. "
                "Run training/4_train_shot.py first."
            )
        self.model          = YOLO(_TRAINED_WEIGHTS)
        self.ball_conf      = ball_conf
        self.hoop_conf      = hoop_conf
        self._cooldown_max  = cooldown_frames

        self.ball_pos  = []   # ((cx,cy), frame_count, w, h, conf)
        self.hoop_pos  = []   # ((cx,cy), frame_count, w, h, conf)

        self.frame_count = 0
        self.makes       = 0
        self.attempts    = 0

        # State
        self._state         = "WATCHING"   # WATCHING | ARMED | COOLDOWN
        self._cooldown      = 0
        self._armed_frames  = 0            # frames spent in ARMED
        self._armed_hoop    = None         # hoop snapshot taken when ARMED
        self._armed_case    = None         # 'A' (ball above rim) | 'B' (ball in bbox from below)
        self._armed_start_frame = 0        # frame_count when ARMED was entered
        self._armed_in_bbox_count = 0      # how many ball detections inside hoop bbox while ARMED
        self._last_ball_frame   = 0        # frame_count when ball was last detected while ARMED
        self._last_ball_in_bbox = False    # was ball inside hoop bbox on last detection while ARMED

    # ------------------------------------------------------------------ #

    def update(self, frame: np.ndarray) -> dict:
        """
        Returns dict:
          scored    : bool
          attempt   : bool
          make      : bool
          makes     : int
          attempts  : int
          state     : "WATCHING" | "ARMED" | "COOLDOWN"
          ball_bbox : (x1,y1,x2,y2) | None
          hoop_bbox : (x1,y1,x2,y2) | None
        """
        result = {
            "scored":    False,
            "attempt":   False,
            "make":      False,
            "makes":     self.makes,
            "attempts":  self.attempts,
            "state":     self._state,
            "ball_bbox": None,
            "hoop_bbox": None,
        }

        # ── Cooldown tick ─────────────────────────────────────────────── #
        if self._cooldown > 0:
            self._cooldown -= 1
            if self._cooldown == 0:
                self._state = "WATCHING"

        # ── YOLO inference ────────────────────────────────────────────── #
        preds = self.model(frame, conf=min(self.ball_conf, self.hoop_conf),
                           verbose=False)[0]

        for box in preds.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            if cls == _CLS_BALL:
                near = in_hoop_region(center, self.hoop_pos)
                if conf >= (0.15 if near else self.ball_conf):
                    self.ball_pos.append((center, self.frame_count, w, h, conf))
                    result["ball_bbox"] = (x1, y1, x2, y2)

            elif cls == _CLS_HOOP and conf >= self.hoop_conf:
                self.hoop_pos.append((center, self.frame_count, w, h, conf))
                result["hoop_bbox"] = (x1, y1, x2, y2)

        # ── Clean positions ────────────────────────────────────────────── #
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count, self.hoop_pos)
        if len(self.hoop_pos) > 1:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)

        # ── State machine ─────────────────────────────────────────────── #
        if self._state == "COOLDOWN":
            pass  # just ticking down

        elif self._state == "WATCHING":
            if self.hoop_pos and self.ball_pos:
                if self._ball_above_rim():
                    # Case A: ball arcing over rim (standard overhand shot)
                    self._state             = "ARMED"
                    self._armed_case        = 'A'
                    self._armed_frames      = 0
                    self._armed_hoop        = list(self.hoop_pos)
                    self._armed_start_frame = self.frame_count
                    self._armed_in_bbox_count = 0
                    self._last_ball_frame   = self.frame_count
                    self._last_ball_in_bbox = False
                    print(f"[SHOT] → ARMED (case A) frame={self.frame_count}")
                elif self._ball_inside_hoop_lower_moving():
                    # Case B: ball rising from below through the hoop
                    self._state             = "ARMED"
                    self._armed_case        = 'B'
                    self._armed_frames      = 0
                    self._armed_hoop        = list(self.hoop_pos)
                    self._armed_start_frame = self.frame_count
                    self._armed_in_bbox_count = 1  # already inside bbox when arming
                    self._last_ball_frame   = self.frame_count
                    self._last_ball_in_bbox = True   # already inside bbox when arming
                    print(f"[SHOT] → ARMED (case B) frame={self.frame_count}")

        elif self._state == "ARMED":
            self._armed_frames += 1

            # Track last ball position while ARMED (for vanish/dunk detection)
            if self.ball_pos:
                bx, by = self.ball_pos[-1][0]
                self._last_ball_frame   = self.frame_count
                in_bbox = self._is_inside_hoop_bbox(bx, by)
                self._last_ball_in_bbox = in_bbox
                if in_bbox:
                    self._armed_in_bbox_count += 1

            frames_since_ball = self.frame_count - self._last_ball_frame

            # Make check 1: ball crossed the rim plane downward (arcing shot)
            if self.hoop_pos and self.ball_pos and self._armed_frames >= 2 and self._ball_crossed_rim():
                self.makes    += 1
                self.attempts += 1
                result["scored"]   = True
                result["attempt"]  = True
                result["make"]     = True
                result["makes"]    = self.makes
                result["attempts"] = self.attempts
                self._state        = "COOLDOWN"
                self._cooldown     = self._cooldown_max
                self._armed_hoop   = None
                self._armed_case   = None
                print(f"[SHOT] MAKE (crossing) frame={self.frame_count}  makes={self.makes}")

            # Make check 2: ball vanished inside hoop bbox (dunk — ball lost in net)
            # Applies regardless of how ARMED was triggered.
            elif (self._last_ball_in_bbox
                  and self._armed_in_bbox_count >= 1
                  and frames_since_ball >= _BBOX_VANISH_FRAMES):
                self.makes    += 1
                self.attempts += 1
                result["scored"]   = True
                result["attempt"]  = True
                result["make"]     = True
                result["makes"]    = self.makes
                result["attempts"] = self.attempts
                self._state        = "COOLDOWN"
                self._cooldown     = self._cooldown_max
                self._armed_hoop   = None
                self._armed_case   = None
                print(f"[SHOT] MAKE (vanish) frame={self.frame_count}  makes={self.makes}")

            elif self._armed_frames > _ARMED_TIMEOUT:
                # Ball never scored — count as miss and reset
                self.attempts += 1
                result["attempt"]  = True
                result["make"]     = False
                result["attempts"] = self.attempts
                self._state      = "WATCHING"
                self._armed_hoop = None
                self._armed_case = None
                print(f"[SHOT] MISS (timeout) frame={self.frame_count}  attempts={self.attempts}")

        result["state"] = self._state
        self.frame_count += 1
        return result

    # ------------------------------------------------------------------ #

    def _rim_bounds(self, hoop_list=None):
        """Return (rim_top_y, x1, x2) from the given hoop list (or current)."""
        hp = hoop_list if hoop_list else self.hoop_pos
        hx, hy = hp[-1][0]
        hw, hh = hp[-1][2], hp[-1][3]
        rim_top = hy - 0.5 * hh
        margin  = _HOOP_H_MARGIN * hw
        return rim_top, hx - 0.5 * hw - margin, hx + 0.5 * hw + margin

    def _ball_moving(self) -> bool:
        """
        True when the ball has moved at least _MIN_BALL_MOTION_PX pixels across
        the last few detections.  Filters out stationary false-positives such as
        the backboard target square being detected as a basketball.
        """
        n = min(5, len(self.ball_pos))
        if n < 2:
            return False
        recent = self.ball_pos[-n:]
        x0, y0 = recent[0][0]
        x1, y1 = recent[-1][0]
        return abs(x1 - x0) + abs(y1 - y0) >= _MIN_BALL_MOTION_PX

    def _ball_above_rim(self) -> bool:
        """
        Case A trigger: ball is above the rim plane within horizontal hoop
        bounds AND is moving.  The movement check prevents false arms on
        stationary objects always above the rim (subtitle text, backboard
        target squares, etc.) being misdetected as basketballs.
        """
        rim_top, x1, x2 = self._rim_bounds()
        bx, by = self.ball_pos[-1][0]
        return by < rim_top and x1 <= bx <= x2 and self._ball_moving()

    def _ball_rose_from_below(self, rim_top: float, curr_bx: int) -> bool:
        """
        True if a recent ball_pos entry at roughly the same horizontal
        position (same object, not a different ball) was at or below rim_top.
        Looks back up to 25 entries.
        """
        for i in range(2, min(26, len(self.ball_pos) + 1)):
            px, py = self.ball_pos[-i][0]
            if abs(px - curr_bx) < 120 and py >= rim_top:
                return True
        return False

    def _ball_inside_hoop_lower_moving(self) -> bool:
        """
        Case B trigger: ball is inside the LOWER portion of the hoop bounding
        box AND is moving upward — characteristic of a dunk or underhand shot
        where the player is below the hoop and the ball rises through it.

        The ball must be in the lower 60 % of the bbox vertically while moving
        toward smaller y (upward in pixel coordinates).  No minimum-distance
        motion gate here — near-rim entry can be very slow.
        """
        if not self.hoop_pos or len(self.ball_pos) < 2:
            return False
        hp = self.hoop_pos[-1]
        hx, hy = hp[0]
        hw, hh = hp[2], hp[3]
        margin = _HOOP_H_MARGIN * hw
        bx, by = self.ball_pos[-1][0]
        bbox_top        = hy - 0.5 * hh
        bbox_bottom     = hy + 0.5 * hh
        lower_threshold = bbox_top + 0.4 * hh   # lower 60 % of bbox

        if not (hx - 0.5 * hw - margin <= bx <= hx + 0.5 * hw + margin
                and lower_threshold <= by <= bbox_bottom):
            return False

        # Ball must be moving upward (y decreasing)
        return self.ball_pos[-1][0][1] < self.ball_pos[-2][0][1]

    def _is_inside_hoop_bbox(self, bx: int, by: int) -> bool:
        """
        True if (bx, by) is inside the CURRENT hoop's full bounding box.
        Uses current hoop_pos so the reference tracks the moving hoop during
        fast plays (dunks, camera panning).
        """
        hp = self.hoop_pos if self.hoop_pos else self._armed_hoop
        if not hp:
            return False
        hx, hy = hp[-1][0]
        hw, hh = hp[-1][2], hp[-1][3]
        margin = _HOOP_H_MARGIN * hw
        return (hx - 0.5 * hw - margin <= bx <= hx + 0.5 * hw + margin
                and hy - 0.5 * hh <= by <= hy + 0.5 * hh)

    def _ball_crossed_rim(self) -> bool:
        """
        Scan recent ball positions for a downward crossing of the rim plane
        while inside horizontal hoop bounds.

        Only considers entries from AFTER the ARMED state was entered —
        stale crossings from previous shots must not trigger a make.
        """
        hp = self.hoop_pos if self.hoop_pos else self._armed_hoop
        rim_top, x1, x2 = self._rim_bounds(hp)

        for i in range(len(self.ball_pos) - 1):
            # Skip entries from before this ARMED cycle
            if self.ball_pos[i][1] < self._armed_start_frame:
                continue
            y_prev = self.ball_pos[i][0][1]
            y_curr = self.ball_pos[i + 1][0][1]
            cx     = self.ball_pos[i + 1][0][0]
            # Ball moves downward across the rim plane within horizontal bounds
            if y_prev < rim_top <= y_curr and x1 <= cx <= x2:
                return True
        return False
