"""
Net motion detector using dense optical flow.

When a basketball passes through the net (swish or bank shot):
  1. The net is pushed DOWNWARD  (ball entering)
  2. The net FLARES / oscillates  (ball exits, net bounces back)

Both phases produce a spike in optical-flow magnitude within the net ROI,
with a predominantly downward direction vector on entry.  No training data
is required — this is purely physics-based temporal analysis.

Net ROI definition
------------------
Given the rim detector's tight bbox (x1, y1, x2, y2) around the orange
ring:

  rim_w      = x2 - x1
  net_height = rim_w * net_height_scale  (net ≈ same length as rim diameter)
  h_pad      = rim_w * h_pad_scale

  net_roi = (x1 - h_pad, y2, x2 + h_pad, y2 + net_height)

The NBA/FIBA net is ~45 cm long; the rim outer diameter is ~46 cm, so a
scale of 1.1–1.3 captures the full net without including too much floor.

Algorithm
---------
1. Convert current and previous frame to grayscale.
2. Crop both to the net ROI.
3. Dense Farneback optical flow in the net crop.
4. Compute:
     mean_mag   – mean magnitude of all flow vectors in the crop
     down_ratio – fraction of pixels where vy > DOWN_THRESH px/frame
5. Update a rolling baseline (median of mean_mag) on quiet frames only
   (frames where the ball bounding circle does not overlap the net ROI).
6. rel_motion = mean_mag / (baseline + eps)
7. State machine:
     WATCHING → COOLDOWN  (scored = True) when
       rel_motion >= trigger_ratio  AND
       down_ratio >= down_ratio_min
       for min_trigger_frames consecutive frames.
     COOLDOWN → WATCHING  after cooldown_frames.

Release-aware sensitivity
-------------------------
Call notify_release() whenever the release detector fires.  For the next
release_window frames the trigger_ratio is temporarily relaxed by
sensitivity_boost to catch the scoring event more reliably.
"""

import cv2
import numpy as np
from collections import deque

# ---------------------------------------------------------------------------
# Farneback optical-flow parameters
# ---------------------------------------------------------------------------
_FB_PARAMS = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=13,
    iterations=3,
    poly_n=5,
    poly_sigma=1.1,
    flags=0,
)

# Minimum vy to count as "downward" motion
_DOWN_THRESH = 0.5  # px/frame


class NetMotionDetector:
    """
    Detects a made basketball shot by monitoring optical flow in the net
    region below the rim.

    Usage
    -----
    detector = NetMotionDetector()

    # once per frame:
    result = detector.update(frame, rim_result, ball_result)
    if result["scored"]:
        print(f"SCORE!  confidence={result['confidence']:.2f}")

    # when release_detector fires:
    detector.notify_release()
    """

    WATCHING = "WATCHING"
    COOLDOWN = "COOLDOWN"

    def __init__(
        self,
        rim_skip_frac:       float = 0.25,  # top fraction of bbox to skip (orange ring hardware)
        x_shrink_frac:       float = 0.05,  # inward horizontal shrink fraction (avoid bracket edges)
        trigger_ratio:       float = 1.7,   # rel_motion threshold (× short-term baseline)
        down_ratio_min:      float = 0.0,   # DISABLED — down_ratio is ~0.04-0.09 at this res/angle
        min_trigger_frames:  int   = 1,     # single frame above threshold is enough
        cooldown_frames:     int   = 80,    # ~2.7 s cooldown — prevents net-oscillation retrigger
        baseline_window:     int   = 5,     # SHORT-TERM window — only compare vs recent quiet frames
        min_net_area:        int   = 400,   # px² — skip if net ROI is degenerate
        sensitivity_boost:   float = 0.7,   # lower the trigger by this after a release
        release_window:      int   = 90,    # frames to stay sensitive after a release
        debug:               bool  = True,  # print per-frame diagnostics
    ):
        self.rim_skip_frac      = rim_skip_frac
        self.x_shrink_frac      = x_shrink_frac
        self.trigger_ratio      = trigger_ratio
        self.down_ratio_min     = down_ratio_min
        self.min_trigger_frames = min_trigger_frames
        self.cooldown_frames    = cooldown_frames
        self.min_net_area       = min_net_area
        self.sensitivity_boost  = sensitivity_boost
        self.release_window     = release_window
        self.debug              = debug
        self.baseline_window    = baseline_window
        # NOTE: net_height_scale and h_pad_scale removed — the rim detector's bbox
        # already wraps the full rim+net assembly, so the net lives *inside* the bbox.

        self._prev_gray: np.ndarray | None = None

        self._state         = self.WATCHING
        self._trigger_count = 0
        self._cooldown_cnt  = 0

        # Rolling baseline for "quiet net" motion level
        self._baseline_buf  = deque(maxlen=baseline_window)
        self._baseline      = 1.0   # small positive seed so we never divide by zero

        # Fixed net ROI — set once when the rim first locks, never moves.
        # The rim bbox jumps erratically frame-to-frame (YOLO noise + tracker drift).
        # If the ROI shifts, the frame-diff measures window movement, not net movement.
        # Freezing it guarantees every pixel comparison is of the same real-world patch.
        self._fixed_net_roi: "tuple | None" = None

        # Release-aware sensitivity counter
        self._release_frames_left = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        frame:       np.ndarray,
        rim_result:  "dict | None",
        ball_result: "tuple | None" = None,  # (cx, cy, r) from BallDetector
    ) -> dict:
        """
        Parameters
        ----------
        frame       : BGR numpy array (current video frame)
        rim_result  : dict from RimDetector.detect_rim(), or None
        ball_result : (cx, cy, r) from BallDetector.detect_ball(), or None

        Returns
        -------
        dict  {
          "scored"     : bool   — True on the single frame a score is confirmed
          "confidence" : float  — 0–1 certainty estimate
          "state"      : str    — "WATCHING" | "COOLDOWN"
          "rel_motion" : float  — motion relative to baseline (>1 = above baseline)
          "down_ratio" : float  — fraction of downward flow vectors
          "net_roi"    : tuple  — (x1,y1,x2,y2) of the net ROI in frame pixels, or None
        }
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Tick the release-sensitivity countdown
        if self._release_frames_left > 0:
            self._release_frames_left -= 1

        # First frame: no previous frame available
        if self._prev_gray is None:
            self._prev_gray = gray
            return self._make_result(False, 0.0, 0.0, 0.0, None)

        # No rim locked → cannot define the net ROI
        if rim_result is None:
            self._prev_gray = gray
            return self._make_result(False, 0.0, 0.0, 0.0, None)

        # Freeze the net ROI on the first valid locked detection.
        # Never update it while running — the YOLO bbox jitters each frame and
        # moving the crop window creates artificial frame-diff even when the net
        # is perfectly still.  Only reset() clears this.
        if self._fixed_net_roi is None:
            if rim_result.get("locked", False):
                candidate = self._compute_net_roi(rim_result, frame.shape)
                if candidate is not None:
                    cw = candidate[2] - candidate[0]
                    ch = candidate[3] - candidate[1]
                    if cw * ch >= self.min_net_area:
                        self._fixed_net_roi = candidate

        net_roi = self._fixed_net_roi
        if net_roi is None:
            self._prev_gray = gray
            return self._make_result(False, 0.0, 0.0, 0.0, None)

        x1, y1, x2, y2 = net_roi

        # Dense optical flow on the net crop
        prev_crop = self._prev_gray[y1:y2, x1:x2]
        curr_crop = gray[y1:y2, x1:x2]

        # --- Frame-difference signal (not diluted by quiet pixels) ---
        # The ball disturbs only a localised patch of the net; mean magnitude
        # averages that patch into the whole crop and loses it.  The 90th-
        # percentile of |frame_diff| picks up precisely that localised burst.
        diff    = cv2.absdiff(prev_crop, curr_crop).astype(np.float32)
        p90_mag = float(np.percentile(diff, 90))

        # --- Directional signal via optical flow (downward entry) ---
        flow = cv2.calcOpticalFlowFarneback(
            prev_crop, curr_crop, None, **_FB_PARAMS
        )
        fy         = flow[..., 1]
        down_ratio = float(np.mean(fy > _DOWN_THRESH))

        # Check whether the ball overlaps the net ROI (don't pollute baseline)
        ball_in_roi = False
        if ball_result is not None:
            bx, by, br = ball_result
            ball_in_roi = (
                x1 - br < bx < x2 + br and
                y1 - br < by < y2 + br
            )

        # Update baseline on all ball-free frames (including during COOLDOWN).
        # Keeping the baseline live during COOLDOWN prevents the post-shot player
        # movement from leaving it stale — which would otherwise cause a false
        # re-trigger the moment COOLDOWN expires.
        if not ball_in_roi:
            self._baseline_buf.append(p90_mag)
            if self._baseline_buf:
                self._baseline = float(np.median(self._baseline_buf)) + 0.01

        rel_motion = p90_mag / (self._baseline + 1e-6)

        # Effective trigger ratio (relaxed for a window after a release)
        effective_trigger = (
            self.trigger_ratio * self.sensitivity_boost
            if self._release_frames_left > 0
            else self.trigger_ratio
        )

        if self.debug:
            warmup = len(self._baseline_buf)
            print(
                f"[NET] state={self._state:8s} | "
                f"p90={p90_mag:.1f} | "
                f"base={self._baseline:.1f} | "
                f"rel={rel_motion:.2f}x | "
                f"dr={down_ratio:.2f} | "
                f"trig={effective_trigger:.2f} | "
                f"warmup={warmup}/{self.baseline_window} | "
                f"cnt={self._trigger_count}"
            )

        # ------------------------------------------------------------------
        # State machine
        # ------------------------------------------------------------------
        scored     = False
        confidence = 0.0

        if self._state == self.WATCHING:
            # Don't allow triggering until we have a real baseline — the first
            # few frames always produce artificially high rel_motion because
            # self._baseline is initialised to 1.0 before any data is collected.
            baseline_ready = len(self._baseline_buf) >= self.baseline_window

            above = baseline_ready and rel_motion >= effective_trigger
            if above:
                self._trigger_count += 1
            else:
                self._trigger_count = 0

            if self._trigger_count >= self.min_trigger_frames:
                scored              = True
                confidence          = self._score_confidence(rel_motion, down_ratio)
                self._state         = self.COOLDOWN
                self._cooldown_cnt  = self.cooldown_frames
                self._trigger_count = 0

        elif self._state == self.COOLDOWN:
            self._cooldown_cnt -= 1
            if self._cooldown_cnt <= 0:
                self._state = self.WATCHING

        self._prev_gray = gray
        result = self._make_result(scored, confidence, rel_motion, down_ratio, net_roi)
        return result

    def notify_release(self):
        """
        Call this whenever the release detector fires.
        For the next `release_window` frames the trigger threshold is
        relaxed by `sensitivity_boost` so a scoring net-motion event right
        after a release is caught more reliably.
        """
        self._release_frames_left = self.release_window

    def reset(self):
        """Full reset — call when switching to a new video / camera cut."""
        self._prev_gray           = None
        self._fixed_net_roi       = None
        self._state               = self.WATCHING
        self._trigger_count       = 0
        self._cooldown_cnt        = 0
        self._baseline_buf.clear()
        self._baseline            = 1.0
        self._release_frames_left = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_net_roi(
        self,
        rim_result: dict,
        frame_shape: tuple,
    ) -> "tuple | None":
        """
        Return (x1, y1, x2, y2) clamped to frame bounds, or None.

        The rim detector's bbox wraps the *entire* rim+net assembly:
          - Top portion  (~top 25%)  = orange steel ring + bracket hardware
          - Lower portion (~bottom 75%) = the net itself

        We therefore sample *inside* the bbox, skipping the top `rim_skip_frac`
        to land squarely in the net region.
        """
        H, W   = frame_shape[:2]
        rx1, ry1, rx2, ry2 = rim_result["bbox"]
        bbox_h = ry2 - ry1
        bbox_w = rx2 - rx1

        if bbox_h < 10 or bbox_w < 10:
            return None

        # Skip the top band (orange rim ring hardware)
        rim_skip = int(bbox_h * self.rim_skip_frac)
        # Slight inward horizontal shrink to avoid bracket/backboard edges
        x_shrink = int(bbox_w * self.x_shrink_frac)

        x1 = max(0, rx1 + x_shrink)
        x2 = min(W, rx2 - x_shrink)
        y1 = max(0, ry1 + rim_skip)   # start below orange ring
        y2 = min(H, ry2)              # end at bottom of bbox (bottom of net)

        if x2 <= x1 or y2 <= y1:
            return None

        return (x1, y1, x2, y2)

    def _score_confidence(self, rel_motion: float, down_ratio: float) -> float:
        """Blend relative-motion and down-ratio into a 0–1 confidence value."""
        # rel_motion: trigger threshold = 0.0, 6× baseline = 1.0
        motion_score = np.clip(
            (rel_motion - self.trigger_ratio) / max(6.0 - self.trigger_ratio, 1e-6),
            0.0, 1.0,
        )
        # down_ratio: 0.38 threshold → 0.0, 0.70 = 1.0
        dir_score = np.clip(
            (down_ratio - self.down_ratio_min) / max(0.70 - self.down_ratio_min, 1e-6),
            0.0, 1.0,
        )
        return round(float(0.6 * motion_score + 0.4 * dir_score), 3)

    def _make_result(
        self,
        scored:     bool,
        confidence: float,
        rel_motion: float,
        down_ratio: float,
        net_roi:    "tuple | None",
    ) -> dict:
        return {
            "scored":     scored,
            "confidence": round(confidence, 3),
            "rel_motion": round(rel_motion, 3),
            "down_ratio": round(down_ratio, 3),
            "net_roi":    net_roi,
            "state":      self._state,
        }
