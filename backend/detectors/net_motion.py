"""
Net-motion make/miss detector.

Instead of tracking the ball through the rim (unreliable with noisy YOLO),
we watch the **net itself**.  A made shot causes visible net motion (swish,
ripple); a miss doesn't.

How it works
────────────
1. Each frame, extract a ROI just below the rim bbox (where the net hangs).
2. Compute per-pixel absolute difference between current and previous ROI.
3. Subtract global frame motion (camera shake compensation).
4. The resulting "net score" is a single float: high = net moved, low = still.
5. Maintain a rolling baseline of net scores during quiet periods.
6. When the ShotDetector goes ARMED (ball near rim), start watching.
   - If net score spikes > threshold above baseline → MAKE
   - If timeout expires with no spike → MISS

The detector is **completely independent of ball tracking** near the rim.
It only needs the hoop bbox (which is rock-solid) and the frame pixels.
"""

import cv2
import numpy as np
from collections import deque


class NetMotionDetector:
    """
    Per-frame net motion scorer.

    Call ``update(frame, hoop_bbox, armed)`` every frame.

    Parameters
    ----------
    spike_thresh : float
        A net motion reading must exceed ``baseline * spike_thresh`` to
        count as a make.  Higher = more conservative.
    confirm_frames : int
        Number of consecutive above-threshold frames required to confirm
        a make.  Prevents one-frame flukes.
    decision_window : int
        After ARMED, how many frames to watch before declaring a miss.
    cooldown_frames : int
        Frames to ignore after a make/miss decision (prevents double-count).
    baseline_window : int
        Rolling window size for baseline net-motion estimation.
    """

    IDLE     = "IDLE"
    WATCHING = "WATCHING"
    DECIDED  = "DECIDED"

    def __init__(
        self,
        spike_thresh:    float = 3.0,
        min_peak:        float = 8.0,
        confirm_frames:  int   = 2,
        make_window:     int   = 25,
        decision_window: int   = 50,
        cooldown_frames: int   = 75,
        baseline_window: int   = 60,
        self_arm_thresh: float = 2.5,
        debug:           bool  = True,
    ):
        self.spike_thresh    = spike_thresh
        self.min_peak        = min_peak
        self.confirm_frames  = confirm_frames
        self.make_window     = make_window
        self.decision_window = decision_window
        self.cooldown_frames = cooldown_frames
        self.self_arm_thresh = self_arm_thresh
        self.debug           = debug

        # Rolling baseline of net motion during quiet (non-armed) periods
        self._baseline_buf = deque(maxlen=baseline_window)

        # Previous frame's net ROI (grayscale, resized to standard size)
        self._prev_net_roi  = None
        # Previous full frame grayscale (for camera shake estimation)
        self._prev_gray     = None

        # State
        self._state          = self.IDLE
        self._watch_count    = 0       # frames spent in WATCHING
        self._spike_count    = 0       # consecutive above-threshold frames
        self._cooldown_cnt   = 0
        self._peak_score     = 0.0     # highest net score during this watch window

        # Counters
        self.makes    = 0
        self.attempts = 0

        # Last decision info (for HUD)
        self.last_decision      = None   # "make" | "miss" | None
        self.last_decision_conf = 0.0

    # ── public API ────────────────────────────────────────────────────── #

    def update(self, frame: np.ndarray, hoop_bbox, armed: bool) -> dict:
        """
        Call every frame.

        Parameters
        ----------
        frame : BGR frame (640×480 display resolution).
        hoop_bbox : (x1, y1, x2, y2) of the hoop in frame coords, or None.
        armed : True when the upstream ShotDetector is in ARMED state
                (ball detected near the rim — a shot attempt is in progress).

        Returns
        -------
        dict with keys:
            attempt  : bool — a decision was just made (make or miss)
            make     : bool — the decision was "make"
            makes    : int  — cumulative makes
            attempts : int  — cumulative attempts
            net_score: float — raw net motion score this frame
            state    : str  — current state
        """
        result = {
            "attempt":   False,
            "make":      False,
            "makes":     self.makes,
            "attempts":  self.attempts,
            "net_score": 0.0,
            "state":     self._state,
        }

        # ── Cooldown tick ─────────────────────────────────────────── #
        if self._cooldown_cnt > 0:
            self._cooldown_cnt -= 1
            if self._cooldown_cnt == 0:
                self._state = self.IDLE
            # Still compute net score for baseline building, but don't decide
            net_score = self._compute_net_score(frame, hoop_bbox)
            if net_score is not None:
                result["net_score"] = net_score
            result["state"] = self._state
            return result

        # ── Compute net motion score ──────────────────────────────── #
        net_score = self._compute_net_score(frame, hoop_bbox)
        if net_score is not None:
            result["net_score"] = net_score

        # ── Baseline update (only when idle — net should be still) ── #
        if self._state == self.IDLE and net_score is not None:
            self._baseline_buf.append(net_score)

        baseline = self._get_baseline()

        # ── State transitions ─────────────────────────────────────── #
        if self._state == self.IDLE:
            # Arm from upstream ShotDetector OR self-arm when net motion
            # spikes above baseline (handles case where ShotDetector is
            # unavailable — e.g. no trained weights on this machine).
            should_arm = armed
            if not should_arm and net_score is not None and baseline > 0:
                if net_score > baseline * self.self_arm_thresh and net_score >= self.min_peak * 0.5:
                    should_arm = True
            if should_arm:
                self._state       = self.WATCHING
                self._watch_count = 0
                self._spike_count = 0
                self._peak_score  = 0.0
                if self.debug:
                    src = "ARMED" if armed else "SELF-ARM"
                    print(f"[NET] → WATCHING ({src})  baseline={baseline:.1f}  score={net_score}")

        elif self._state == self.WATCHING:
            self._watch_count += 1

            if net_score is not None:
                if net_score > self._peak_score:
                    self._peak_score = net_score

                threshold = max(baseline * self.spike_thresh, baseline + 4.0)

                if net_score > threshold:
                    self._spike_count += 1
                else:
                    # Allow small gaps — don't reset immediately
                    if self._spike_count > 0:
                        self._spike_count = max(0, self._spike_count - 1)

                if self.debug and self._watch_count % 5 == 0:
                    print(
                        f"[NET] WATCHING f={self._watch_count:3d} "
                        f"score={net_score:.1f} base={baseline:.1f} "
                        f"thresh={threshold:.1f} spikes={self._spike_count}"
                    )

                # ── MAKE decision ─────────────────────────────────── #
                # Only accept makes within the make_window — real swishes
                # cause immediate net motion. Late spikes are rim bounces.
                if (self._spike_count >= self.confirm_frames
                        and self._peak_score >= self.min_peak
                        and self._watch_count <= self.make_window):
                    self.makes    += 1
                    self.attempts += 1
                    result["attempt"] = True
                    result["make"]    = True
                    result["makes"]   = self.makes
                    result["attempts"] = self.attempts
                    self.last_decision      = "make"
                    self.last_decision_conf = self._peak_score / max(baseline, 1.0)
                    self._state        = self.DECIDED
                    self._cooldown_cnt = self.cooldown_frames
                    if self.debug:
                        print(
                            f"[NET] ★ MAKE  peak={self._peak_score:.1f} "
                            f"baseline={baseline:.1f} "
                            f"ratio={self.last_decision_conf:.1f}x"
                        )

            # ── MISS decision (timeout) ───────────────────────────── #
            if self._state == self.WATCHING and self._watch_count > self.decision_window:
                # Only count as a real miss if peak was significant —
                # a low peak means this was a false ARMED trigger (e.g. a
                # pass near the hoop) rather than an actual shot attempt.
                if self._peak_score >= self.min_peak:
                    self.attempts += 1
                    result["attempt"]  = True
                    result["make"]     = False
                    result["attempts"] = self.attempts
                    self.last_decision      = "miss"
                    self.last_decision_conf = 0.0
                    if self.debug:
                        print(
                            f"[NET] ✗ MISS  peak={self._peak_score:.1f} "
                            f"baseline={baseline:.1f}"
                        )
                else:
                    if self.debug:
                        print(
                            f"[NET] — IGNORED (low peak={self._peak_score:.1f}, "
                            f"likely not a shot)"
                        )
                self._state        = self.DECIDED
                self._cooldown_cnt = self.cooldown_frames

        result["state"] = self._state
        return result

    # ── internals ─────────────────────────────────────────────────────── #

    def _compute_net_score(self, frame: np.ndarray, hoop_bbox) -> float | None:
        """
        Compute net motion score for this frame.

        Returns None if we can't compute (no hoop, no previous frame, etc).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if hoop_bbox is None or self._prev_gray is None:
            self._prev_gray = gray
            return None

        x1, y1, x2, y2 = hoop_bbox
        hoop_h = y2 - y1
        hoop_w = x2 - x1
        fh, fw = frame.shape[:2]

        # Net ROI: from bottom of hoop bbox, extending down 1.5× hoop height
        # Horizontally: slightly narrower than hoop (the net is inside the rim)
        net_y1 = y2
        net_y2 = min(fh, y2 + int(hoop_h * 1.5))
        net_x1 = max(0, x1 + int(hoop_w * 0.1))
        net_x2 = min(fw, x2 - int(hoop_w * 0.1))

        # Sanity: ROI must be at least 10×10
        if net_y2 - net_y1 < 10 or net_x2 - net_x1 < 10:
            self._prev_gray = gray
            return None

        # Extract ROI from current and previous frame
        roi_curr = gray[net_y1:net_y2, net_x1:net_x2]
        roi_prev = self._prev_gray[net_y1:net_y2, net_x1:net_x2]

        # Resize to standard size for consistent scoring across resolutions
        _STD = (48, 48)
        roi_curr_r = cv2.resize(roi_curr, _STD)
        roi_prev_r = cv2.resize(roi_prev, _STD)

        # Per-pixel absolute difference
        diff = cv2.absdiff(roi_curr_r, roi_prev_r).astype(np.float32)
        net_motion = float(np.mean(diff))

        # Camera shake compensation: compute global frame diff
        global_diff = cv2.absdiff(gray, self._prev_gray).astype(np.float32)
        global_motion = float(np.mean(global_diff))

        # Net-local motion: use RATIO-based scoring so it works at any fps.
        # At low fps (4fps), both net_motion and global_motion are large, so
        # subtraction wipes out real signals.  Ratio preserves them:
        # if the net moved MORE than the rest of the frame, the ratio > 1.
        if global_motion > 0.5:
            net_score = max(0.0, (net_motion / global_motion) - 1.0) * global_motion
        else:
            # Very low global motion (static camera) — use raw difference
            net_score = net_motion

        self._prev_gray = gray
        return net_score

    def _get_baseline(self) -> float:
        """Return the current baseline net motion (median of recent idle scores)."""
        if len(self._baseline_buf) < 5:
            return 2.0   # sensible default before we have data
        return float(np.median(self._baseline_buf))
