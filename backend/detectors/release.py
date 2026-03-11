"""
Pose-based basketball release detector.

Detects the shooting pose — wrist above shoulder with elbow extended.
Uses a sliding window: fires when *confirm_hits* of the last *window_size*
detect() calls had the shooting pose.  This handles occasional dropped
frames from the pose detector without requiring strict consecutive hits.
"""

from collections import deque


class ReleaseDetector:

    def __init__(
        self,
        elbow_threshold:  float = 130.0,
        confirm_hits:     int   = 2,
        window_size:      int   = 5,
        cooldown_frames:  int   = 60,
        debug:            bool  = True,
    ):
        self.elbow_threshold = elbow_threshold
        self.confirm_hits    = confirm_hits
        self.window_size     = window_size
        self.cooldown_frames = cooldown_frames
        self.debug           = debug

        self._state        = "IDLE"     # IDLE | COOLDOWN
        self._cooldown_cnt = 0
        self._window       = deque(maxlen=window_size)

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

        # ── Cooldown ──────────────────────────────────────────────── #
        if self._state == "COOLDOWN":
            self._cooldown_cnt -= 1
            if self._cooldown_cnt <= 0:
                self._state = "IDLE"
            self._window.clear()
            return self._result(False, 0.0)

        # ── Shooting pose check ───────────────────────────────────── #
        shoulder_mid_y = (shoulder_l_y + shoulder_r_y) / 2.0
        wrist_above    = wrist_y < shoulder_mid_y
        elbow_extended = elbow_angle > self.elbow_threshold

        in_pose = wrist_above and elbow_extended
        self._window.append(in_pose)

        hits = sum(self._window)
        is_release = hits >= self.confirm_hits

        if self.debug:
            print(
                f"[RD] {self._state:8s} | "
                f"ea={elbow_angle:6.1f} | "
                f"above={'Y' if wrist_above else 'N'} | "
                f"ext={'Y' if elbow_extended else 'N'} | "
                f"hits={hits}/{len(self._window)}"
            )

        if is_release:
            self._state        = "COOLDOWN"
            self._cooldown_cnt = self.cooldown_frames
            self._window.clear()
            conf = min(1.0, elbow_angle / 180.0)
            print(f"RELEASE DETECTED  confidence={conf:.2f}  ea={elbow_angle:.1f}")
            return self._result(True, conf)

        return self._result(False, 0.0)

    def reset(self):
        self._state        = "IDLE"
        self._cooldown_cnt = 0
        self._window.clear()

    # ------------------------------------------------------------------

    def _result(self, release: bool, confidence: float) -> dict:
        return {
            "release":    release,
            "confidence": round(confidence, 4),
            "state":      self._state,
            "signals":    {},
        }