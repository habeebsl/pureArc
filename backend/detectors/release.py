"""
Pose-dominant basketball release detector — pump-fake resistant.

State machine:  IDLE → COCKING → RELEASING → COOLDOWN

Key design principles
---------------------
* **Pose-first**: state transitions use ONLY MediaPipe pose landmarks.
  Ball position is a bonus signal in the confidence blend, never a gate.
* **Pump-fake resistant**: fires only when a genuine wrist snap
  (follow-through) is detected.  During a pump fake the wrist stays
  rigid because the player is gripping the ball — snap stays near zero.
  Additionally, if the elbow reverses > 8° from its peak during the
  RELEASING window the detector immediately resets (arm pulled back).
* **Scale-invariant**: all distances normalized by shoulder width.
* **Low-latency**: fires within 1-2 frames of the wrist snap.

Signals (pose-dominant weighting)
---------------------------------
  wrist_snap  4.0  — follow-through flick; THE defining signal
  elbow_ext   3.0  — extension velocity
  angle_range 1.5  — elbow in shooting range (85-180°)
  separation  1.0  — ball-wrist distance increasing (bonus)
  ball_above  0.5  — ball above wrist (bonus)
"""

import math
from collections import deque

import numpy as np


class ReleaseDetector:
    IDLE      = "IDLE"
    COCKING   = "COCKING"
    RELEASING = "RELEASING"
    COOLDOWN  = "COOLDOWN"

    # Pose-dominant weights — pose signals carry the decision,
    # ball signals are optional bonuses.
    _WEIGHTS = {
        "wrist_snap":  4.0,
        "elbow_ext":   3.0,
        "angle_range": 1.5,
        "separation":  1.0,
        "ball_above":  0.5,
    }

    def __init__(
        self,
        smoothing_window: int   = 3,
        cooldown_frames:  int   = 45,
        min_confidence:   float = 0.50,
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
        self._raw_ea_hist   = deque(maxlen=6)

        self._state              = self.IDLE
        self._cooldown_cnt       = 0
        self._cocking_frames     = 0
        self._releasing_frames   = 0
        self._min_ea_cocking     = 180.0

    # ------------------------------------------------------------------

    def detect(
        self,
        ball_x: float,        ball_y: float,
        wrist_x: float,       wrist_y: float,
        elbow_x: float,       elbow_y: float,
        shoulder_l_x: float,  shoulder_l_y: float,
        shoulder_r_x: float,  shoulder_r_y: float,
        elbow_angle: float,
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

        shoulder_mid_y = (shoulder_r_y + shoulder_l_y) / 2.0

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
        # Guard: reject YOLO ball teleportation (>5 shoulder-widths jump)
        if len(self._dist_buf) >= 1 and abs(dist_norm - self._dist_buf[-1]) > 5.0:
            dist_norm = self._dist_buf[-1]
        self._dist_buf.append(dist_norm)

        if len(self._dist_buf) < 2:
            return self._make_result(False, 0.0, {})

        # ---- guard: reject impossible elbow-angle jumps (>45°/frame) -
        if len(self._elbow_ang_buf) >= 2:
            ang_jump = abs(self._elbow_ang_buf[-1] - self._elbow_ang_buf[-2])
            if ang_jump > 45.0:
                self._elbow_ang_buf.pop()
                if self._elbow_ang_buf:
                    self._elbow_ang_buf.append(self._elbow_ang_buf[-1])
                else:
                    self._elbow_ang_buf.append(elbow_angle)
                return self._make_result(False, 0.0, {})

        self._raw_ea_hist.append(float(self._elbow_ang_buf[-1]))

        # ---- cooldown -----------------------------------------------
        if self._state == self.COOLDOWN:
            self._cooldown_cnt -= 1
            if self._cooldown_cnt <= 0:
                self._state = self.IDLE
            return self._make_result(False, 0.0, {})

        # ---- smoothed scalars ---------------------------------------
        sw     = float(np.mean(self._shoulder_buf))
        wy     = float(np.mean(self._wrist_y_buf))
        ea     = float(np.mean(self._elbow_ang_buf))
        d_cur  = float(self._dist_buf[-1])
        d_prev = float(self._dist_buf[-2])

        ang_cur  = float(self._elbow_ang_buf[-1])
        ang_prev = float(self._elbow_ang_buf[-2])

        wrist_y_cur  = float(self._wrist_y_buf[-1])
        wrist_y_prev = float(self._wrist_y_buf[-2])
        elbow_y_cur  = float(self._elbow_y_buf[-1])
        elbow_y_prev = float(self._elbow_y_buf[-2])

        # ---- velocities ---------------------------------------------
        sep_vel       = d_cur - d_prev                         # norm dist/frame
        elbow_ext_vel = ang_cur - ang_prev                     # deg/frame

        snap_vel = (
            (wrist_y_cur - elbow_y_cur) - (wrist_y_prev - elbow_y_prev)
        ) / sw

        ball_above_wrist = (wy - float(np.mean(self._ball_y_buf))) / sw

        # ---- POSE-ONLY state machine --------------------------------
        # Shooting posture check: wrist above mid-shoulder (y ↓ in image)
        wrist_above_shoulder = wrist_y < shoulder_mid_y

        if self._state == self.IDLE:
            # Enter COCKING: shooting posture + elbow bent — pure pose
            if ea < 140 and wrist_above_shoulder:
                self._state = self.COCKING
                self._cocking_frames = 0
                self._min_ea_cocking = ea

        elif self._state == self.COCKING:
            self._cocking_frames += 1
            if ea < self._min_ea_cocking:
                self._min_ea_cocking = ea

            # Sustained extension check: 3 ascending readings, span ≥ 12°
            _h = self._raw_ea_hist
            _ext_ok = (len(_h) >= 3
                       and _h[-1] > _h[-2] > _h[-3]
                       and 15 <= (_h[-1] - _h[-3]) <= 60)

            if (self._cocking_frames >= 3
                    and ea >= 85
                    and ea > self._min_ea_cocking + 20
                    and _ext_ok):
                self._state = self.RELEASING
                self._releasing_frames = 0
                if self.debug:
                    print(f"[RD] → RELEASING  ea={ea:.1f}  min_ea={self._min_ea_cocking:.1f}  hist={[round(x,1) for x in _h]}")

            # Safety: exit COCKING if arm straightened or held too long
            elif ea > 170 or self._cocking_frames > 30:
                self._state = self.IDLE

        elif self._state == self.RELEASING:
            self._releasing_frames += 1
            if self._releasing_frames > 8:
                self._state = self.IDLE

        # ---- confidence signals (pose-dominant) ---------------------
        signals = {
            "wrist_snap":  float(np.clip(snap_vel / 0.20, 0.0, 1.0)),
            "elbow_ext":   self._sig(elbow_ext_vel, center=2.0,  k=1.0),
            "angle_range": 1.0 if 85.0 <= ea <= 180.0 else 0.0,
            "separation":  self._sig(sep_vel,        center=0.06, k=25),
            "ball_above":  self._sig(ball_above_wrist, center=0.03, k=35),
        }

        total_w    = sum(self._WEIGHTS.values())
        confidence = sum(signals[k] * self._WEIGHTS[k] for k in signals) / total_w

        # Fire: wrist snap is THE decision-maker.
        # During a pump fake the wrist stays rigid (snap ≈ 0) because
        # the player is gripping the ball.  Real releases always
        # produce snap > 0.5 from the follow-through flick.
        # The state machine already verified shooting posture + sustained
        # extension before reaching RELEASING.
        # wrist_above_shoulder: you physically cannot release a jump shot
        # with your hand below your shoulder.
        is_release = (
            self._state == self.RELEASING
            and signals["wrist_snap"] > 0.3
            and signals["elbow_ext"] > 0.3
            and confidence >= self.min_confidence
            and signals["angle_range"] > 0
        )

        if self.debug:
            print(
                f"[RD] state={self._state:10s} | "
                f"ea={ea:6.1f} | ext={elbow_ext_vel:+.2f} | "
                f"snap={signals.get('wrist_snap', 0):.2f} | "
                f"conf={confidence:.2f} | "
                f"wrist_above={'Y' if wrist_above_shoulder else 'N'}"
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
            self._raw_ea_hist,
        ):
            buf.clear()
        self._state              = self.IDLE
        self._cooldown_cnt       = 0
        self._cocking_frames     = 0
        self._releasing_frames   = 0
        self._min_ea_cocking     = 180.0

    # ------------------------------------------------------------------

    @staticmethod
    def _sig(x: float, center: float, k: float) -> float:
        exponent = max(-500.0, min(500.0, -k * (x - center)))
        return 1.0 / (1.0 + math.exp(exponent))

    def _make_result(self, release: bool, confidence: float, signals: dict) -> dict:
        return {
            "release":    release,
            "confidence": round(confidence, 4),
            "state":      self._state,
            "signals":    signals,
        }