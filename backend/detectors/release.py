"""
High-quality basketball release detector.

State machine:  IDLE → COCKING → RELEASING → COOLDOWN

IDLE      : waiting for a pre-shot setup
COCKING   : elbow bent (<155°) and ball is in the upper-body zone
RELEASING : elbow extending + ball separating from wrist → score confidence
COOLDOWN  : mandatory gap after a detected release

Key design decisions
---------------------
* Scale-invariant: all distances divided by shoulder width.
* Light 3-frame smoothing (enough to kill jitter, not enough to kill spikes).
* COCKING gate does NOT require the ball detector to have confirmed the ball
  is in the hand — YOLO's bbox centre drifts enough that that gate was
  permanently blocking the state machine.  Instead we enter COCKING as soon
  as the elbow is bent AND the ball is anywhere in the upper-body zone.
* RELEASING fires on elbow extension velocity alone (≥ 0.8 °/frame after
  smoothing) plus ball separation — both thresholds are deliberately loose
  so a real shot always clears them.
* Confidence is a 6-signal weighted sigmoid blend; only the RELEASING phase
  can produce a True detection.
* debug=True prints a single line per frame so you can watch exactly what
  every signal is doing in real time.
"""

import math
from collections import deque

import numpy as np


class ReleaseDetector:
    IDLE      = "IDLE"
    COCKING   = "COCKING"
    RELEASING = "RELEASING"
    COOLDOWN  = "COOLDOWN"

    _WEIGHTS = {
        "separation":  3.0,
        "ball_upward": 2.5,
        "elbow_ext":   2.0,
        "wrist_snap":  1.5,
        "ball_above":  1.0,
        "angle_range": 0.5,
    }

    def __init__(
        self,
        smoothing_window: int   = 3,
        cooldown_frames:  int   = 30,
        min_confidence:   float = 0.55,
        debug:            bool  = True,
    ):
        self.smoothing_window = smoothing_window
        self.cooldown_frames  = cooldown_frames
        self.min_confidence   = min_confidence
        self.debug            = debug

        n = smoothing_window
        self._ball_x_buf    = deque(maxlen=n)
        self._ball_y_buf    = deque(maxlen=n)
        self._wrist_x_buf   = deque(maxlen=n)
        self._wrist_y_buf   = deque(maxlen=n)
        self._elbow_y_buf   = deque(maxlen=n)
        self._elbow_ang_buf = deque(maxlen=n)
        self._shoulder_buf  = deque(maxlen=n)
        self._dist_buf      = deque(maxlen=n)

        self._state          = self.IDLE
        self._cooldown_cnt   = 0
        self._cocking_frames = 0   # consecutive frames in COCKING

    # ------------------------------------------------------------------

    def detect(
        self,
        ball_x: float,        ball_y: float,
        wrist_x: float,       wrist_y: float,
        elbow_x: float,       elbow_y: float,
        shoulder_l_x: float,  shoulder_l_y: float,
        shoulder_r_x: float,  shoulder_r_y: float,
        elbow_angle: float,
        # optional: pass hip_y so we can define "upper body zone" properly
        hip_y: float = None,
        frame_h: float = 480,
    ) -> dict:
        """
        Returns {'release': bool, 'confidence': float, 'state': str, 'signals': dict}
        """

        # ---- shoulder width -----------------------------------------
        shoulder_w = math.hypot(
            shoulder_r_x - shoulder_l_x,
            shoulder_r_y - shoulder_l_y,
        )
        if shoulder_w < 1.0:
            shoulder_w = 1.0

        # ---- mid-shoulder x (used for proximity check) ---------------
        shoulder_mid_x = (shoulder_r_x + shoulder_l_x) / 2.0
        shoulder_mid_y = (shoulder_r_y + shoulder_l_y) / 2.0

        # ---- upper-body zone boundary --------------------------------
        # Ball must be within 2.5 shoulder-widths of shoulder midpoint
        # to be considered "in play" (prevents faraway ball triggering COCKING)
        upper_zone_radius = 2.5 * shoulder_w
        ball_to_shoulder = math.hypot(
            ball_x - shoulder_mid_x,
            ball_y - shoulder_mid_y,
        )
        ball_in_upper_zone = ball_to_shoulder < upper_zone_radius

        # ---- push history -------------------------------------------
        self._ball_x_buf.append(ball_x)
        self._ball_y_buf.append(ball_y)
        self._wrist_x_buf.append(wrist_x)
        self._wrist_y_buf.append(wrist_y)
        self._elbow_y_buf.append(elbow_y)
        self._elbow_ang_buf.append(elbow_angle)
        self._shoulder_buf.append(shoulder_w)

        dist_raw  = math.hypot(ball_x - wrist_x, ball_y - wrist_y)
        dist_norm = dist_raw / shoulder_w
        self._dist_buf.append(dist_norm)

        if len(self._dist_buf) < 2:
            return self._make_result(False, 0.0, {})

        # ---- guard: reject frames with impossible elbow angle jumps --
        # MediaPipe frequently produces garbage single-frame readings
        # (e.g. 179° → 26° → 122°). A real elbow cannot move >45° in
        # one frame at 30 fps, so treat such frames as bad data.
        if len(self._elbow_ang_buf) >= 2:
            ang_jump = abs(self._elbow_ang_buf[-1] - self._elbow_ang_buf[-2])
            if ang_jump > 45.0:
                # Pop the bad reading back out and return neutral result
                self._elbow_ang_buf.pop()
                if self._elbow_ang_buf:
                    self._elbow_ang_buf.append(self._elbow_ang_buf[-1])  # repeat last good
                else:
                    self._elbow_ang_buf.append(elbow_angle)
                return self._make_result(False, 0.0, {})

        # ---- cooldown -----------------------------------------------
        if self._state == self.COOLDOWN:
            self._cooldown_cnt -= 1
            if self._cooldown_cnt <= 0:
                self._state = self.IDLE
            return self._make_result(False, 0.0, {})

        # ---- smoothed scalars ---------------------------------------
        sw     = float(np.mean(self._shoulder_buf))
        wy     = float(np.mean(self._wrist_y_buf))
        ey     = float(np.mean(self._elbow_y_buf))
        ea     = float(np.mean(self._elbow_ang_buf))
        d_cur  = float(self._dist_buf[-1])
        d_prev = float(self._dist_buf[-2])

        ball_y_cur  = float(self._ball_y_buf[-1])
        ball_y_prev = float(self._ball_y_buf[-2])

        ang_cur  = float(self._elbow_ang_buf[-1])
        ang_prev = float(self._elbow_ang_buf[-2])

        wrist_y_cur  = float(self._wrist_y_buf[-1])
        wrist_y_prev = float(self._wrist_y_buf[-2])
        elbow_y_cur  = float(self._elbow_y_buf[-1])
        elbow_y_prev = float(self._elbow_y_buf[-2])

        # ---- velocities ---------------------------------------------
        sep_vel      = d_cur - d_prev                          # norm dist/frame
        ball_vy_norm = (ball_y_cur - ball_y_prev) / sw        # neg = going up
        elbow_ext_vel = ang_cur - ang_prev                     # deg/frame

        snap_vel = (
            (wrist_y_cur - elbow_y_cur) - (wrist_y_prev - elbow_y_prev)
        ) / sw

        ball_above_wrist = (wy - float(np.mean(self._ball_y_buf))) / sw

        # ---- state machine ------------------------------------------
        if self._state == self.IDLE:
            # Enter COCKING: elbow bent + ball in upper-body zone
            # No requirement that YOLO confirmed ball in hand — too unreliable
            if ea < 160 and ball_in_upper_zone:
                self._state = self.COCKING
                self._cocking_frames = 0

        elif self._state == self.COCKING:
            self._cocking_frames += 1

            # Enter RELEASING: elbow starts extending + any ball separation
            if elbow_ext_vel >= 0.8 and sep_vel > 0.01:
                self._state = self.RELEASING

            # Safety: exit COCKING if elbow fully straightened without release
            # or ball left the upper body zone for a sustained period
            elif ea > 170 or (not ball_in_upper_zone and self._cocking_frames > 10):
                self._state = self.IDLE

        # ---- confidence (computed in RELEASING only) ----------------
        signals = {
            "separation":  self._sig(sep_vel,          center=0.06, k=25),
            "ball_upward": self._sig(-ball_vy_norm,     center=0.01, k=50),
            "elbow_ext":   self._sig(elbow_ext_vel,     center=2.0,  k=1.0),
            "wrist_snap":  float(np.clip(snap_vel / 0.03, 0.0, 1.0)),
            "ball_above":  self._sig(ball_above_wrist,  center=0.03, k=35),
            "angle_range": 1.0 if 85.0 <= ea <= 180.0 else 0.0,
        }

        total_w    = sum(self._WEIGHTS.values())
        confidence = sum(signals[k] * self._WEIGHTS[k] for k in signals) / total_w

        is_release = (self._state == self.RELEASING) and (confidence >= self.min_confidence)

        if self.debug:
            print(
                f"[RD] state={self._state:10s} | "
                f"ea={ea:6.1f} | "
                f"d_norm={d_cur:.3f} | "
                f"sep={sep_vel:+.4f} | "
                f"ext={elbow_ext_vel:+.2f} | "
                f"zone={'Y' if ball_in_upper_zone else 'N'} | "
                f"conf={confidence:.2f}"
            )

        if is_release:
            self._state          = self.COOLDOWN
            self._cooldown_cnt   = self.cooldown_frames
            self._cocking_frames = 0

        return self._make_result(is_release, confidence, signals)

    def reset(self):
        for buf in (
            self._ball_x_buf, self._ball_y_buf,
            self._wrist_x_buf, self._wrist_y_buf,
            self._elbow_y_buf, self._elbow_ang_buf,
            self._shoulder_buf, self._dist_buf,
        ):
            buf.clear()
        self._state          = self.IDLE
        self._cooldown_cnt   = 0
        self._cocking_frames = 0

    # ------------------------------------------------------------------

    @staticmethod
    def _sig(x: float, center: float, k: float) -> float:
        return 1.0 / (1.0 + math.exp(-k * (x - center)))

    def _make_result(self, release: bool, confidence: float, signals: dict) -> dict:
        return {
            "release":    release,
            "confidence": round(confidence, 4),
            "state":      self._state,
            "signals":    signals,
        }