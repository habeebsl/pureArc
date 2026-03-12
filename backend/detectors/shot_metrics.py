"""
Shot Metrics Engine — converts frame-level detections into structured
measurements describing a single basketball shot.

Usage:
    engine = ShotMetricsEngine()

    # Every frame:
    engine.feed(frame_idx, landmarks, ball_xy, rim_center, angles)

    # When a release is detected:
    engine.on_release(frame_idx)

    # When make/miss is confirmed:
    metrics = engine.on_result(frame_idx, made=True)
    # → ShotMetrics dataclass

The engine buffers per-frame data and computes metrics over the shot
window (pre-release → post-result).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field


# ── Public data structure ──────────────────────────────────────────────

@dataclass
class ShotMetrics:
    """Structured measurements for a single shot attempt."""

    # Release metrics
    release_angle:   float | None = None   # degrees — ball trajectory at release
    release_height:  float | None = None   # fraction of frame height (0=top, 1=bottom)
    elbow_angle:     float | None = None   # degrees at release frame

    # Arc metrics
    arc_peak:        float | None = None   # highest ball y (pixel y, smaller = higher)
    arc_height_ratio:float | None = None   # peak height above release-rim line / horizontal distance (0-1+)
    arc_symmetry:    float | None = None   # 0-1 (1 = perfectly symmetric)

    # Timing metrics
    knee_elbow_lag:  float | None = None   # frames between knee ext and elbow ext (negative = knee first)
    shot_tempo:      int   | None = None   # frames from set point (wrist lowest) to release

    # Distance
    shot_distance_px:float | None = None   # pixel dist from wrist at release to rim center

    # Stability metrics
    torso_drift:     float | None = None   # horizontal shoulder movement (px) during shot

    # Context
    made:            bool  | None = None
    release_frame:   int   | None = None
    result_frame:    int   | None = None


# ── Frame snapshot ─────────────────────────────────────────────────────

@dataclass
class _FrameData:
    frame_idx:    int
    ball_xy:      tuple | None      # (x_px, y_px)
    rim_center:   tuple | None      # (x_px, y_px)
    wrist_xy:     tuple | None      # (x_px, y_px)
    shoulder_mid: tuple | None      # (x_px, y_px)
    elbow_angle:  float | None
    knee_angle:   float | None
    torso_x:      float | None      # horizontal midpoint of shoulders (px)
    wrist_y_norm: float | None      # wrist y / frame_h (0 = top)


# ── Engine ─────────────────────────────────────────────────────────────

# Landmark indices (MediaPipe numbering)
_R_SHOULDER = 12
_L_SHOULDER = 11
_R_ELBOW    = 14
_R_WRIST    = 16
_R_HIP      = 24
_R_KNEE     = 26

# How many frames before release to keep for timing / stability analysis
_PRE_RELEASE_BUF = 60
# Max frames after release to wait for a result before discarding the shot
_POST_RELEASE_MAX = 120


class ShotMetricsEngine:
    """Collects per-frame data and computes metrics per shot."""

    def __init__(self):
        self._buf: deque[_FrameData] = deque(maxlen=_PRE_RELEASE_BUF + _POST_RELEASE_MAX)
        self._release_idx: int | None = None      # frame_idx of last release
        self._shot_pending = False

    # ── Per-frame feed ────────────────────────────────────────────── #

    def feed(
        self,
        frame_idx: int,
        landmarks,            # 33-element list or None
        ball_xy:  tuple | None,
        rim_center: tuple | None,
        angles: dict | None,  # {"elbow_angle": ..., "knee_angle": ...}
        frame_hw: tuple = (480, 640),
    ):
        """Call every frame with available detections."""
        h, w = frame_hw

        wrist_xy = shoulder_mid = None
        elbow_angle = knee_angle = None
        torso_x = wrist_y_norm = None

        if landmarks is not None:
            rw = landmarks[_R_WRIST]
            ls = landmarks[_L_SHOULDER]
            rs = landmarks[_R_SHOULDER]

            if rw.visibility > 0.3:
                wrist_xy = (rw.x * w, rw.y * h)
                wrist_y_norm = rw.y

            if ls.visibility > 0.3 and rs.visibility > 0.3:
                sx = (ls.x * w + rs.x * w) / 2
                sy = (ls.y * h + rs.y * h) / 2
                shoulder_mid = (sx, sy)
                torso_x = sx

        if angles is not None:
            elbow_angle = angles.get("elbow_angle")
            knee_angle  = angles.get("knee_angle")

        self._buf.append(_FrameData(
            frame_idx=frame_idx,
            ball_xy=ball_xy,
            rim_center=rim_center,
            wrist_xy=wrist_xy,
            shoulder_mid=shoulder_mid,
            elbow_angle=elbow_angle,
            knee_angle=knee_angle,
            torso_x=torso_x,
            wrist_y_norm=wrist_y_norm,
        ))

    # ── Events ────────────────────────────────────────────────────── #

    def on_release(self, frame_idx: int):
        """Call when ReleaseDetector fires."""
        self._release_idx = frame_idx
        self._shot_pending = True

    def on_result(self, frame_idx: int, made: bool) -> ShotMetrics | None:
        """
        Call when net-motion detector confirms make/miss.
        Returns computed ShotMetrics, or None if no release was pending.
        """
        if not self._shot_pending or self._release_idx is None:
            return None

        metrics = self._compute(self._release_idx, frame_idx, made)
        self._shot_pending = False
        self._release_idx = None
        return metrics

    @property
    def pending(self) -> bool:
        return self._shot_pending

    # ── Core computation ──────────────────────────────────────────── #

    def _compute(self, release_frame: int, result_frame: int, made: bool) -> ShotMetrics:
        frames = list(self._buf)
        m = ShotMetrics(made=made, release_frame=release_frame, result_frame=result_frame)

        # Partition buffer into pre-release and post-release
        pre  = [f for f in frames if f.frame_idx <= release_frame]
        post = [f for f in frames if f.frame_idx > release_frame]

        # ── Release-frame snapshot ────────────────────────────────── #
        rel = self._at_frame(frames, release_frame)

        if rel is not None:
            m.elbow_angle = rel.elbow_angle
            m.release_height = rel.wrist_y_norm

        # ── Release angle (ball trajectory) ───────────────────────── #
        m.release_angle = self._calc_release_angle(frames, release_frame)

        # ── Distance ──────────────────────────────────────────────── #
        m.shot_distance_px = self._calc_shot_distance(rel)

        # ── Arc metrics ───────────────────────────────────────────── #
        m.arc_peak = self._calc_arc_peak(post)
        m.arc_height_ratio = self._calc_arc_height_ratio(post, rel)
        m.arc_symmetry = self._calc_arc_symmetry(post, release_frame, result_frame)

        # ── Timing metrics ────────────────────────────────────────── #
        m.knee_elbow_lag = self._calc_knee_elbow_lag(pre)
        m.shot_tempo = self._calc_shot_tempo(pre, release_frame)

        # ── Stability metrics ─────────────────────────────────────── #
        m.torso_drift = self._calc_torso_drift(pre)

        return m

    # ── Individual metric calculators ─────────────────────────────────

    def _at_frame(self, frames, idx):
        """Return the _FrameData closest to frame idx."""
        best, best_d = None, float("inf")
        for f in frames:
            d = abs(f.frame_idx - idx)
            if d < best_d:
                best_d = d
                best = f
        return best

    def _calc_release_angle(self, frames, release_frame) -> float | None:
        """
        Angle of ball trajectory at release, computed from ball positions
        in a small window around the release frame.
        """
        window = [f for f in frames
                  if abs(f.frame_idx - release_frame) <= 4 and f.ball_xy is not None]
        if len(window) < 2:
            return None

        window.sort(key=lambda f: f.frame_idx)
        # Use first and last ball positions in the window
        bx0, by0 = window[0].ball_xy
        bx1, by1 = window[-1].ball_xy

        dx = bx1 - bx0
        dy = by0 - by1  # invert y — up is positive

        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return None

        return math.degrees(math.atan2(dy, abs(dx) + 1e-9))

    @staticmethod
    def _calc_shot_distance(rel_frame: _FrameData | None) -> float | None:
        """Pixel distance from wrist at release to rim center."""
        if rel_frame is None:
            return None
        if rel_frame.wrist_xy is None or rel_frame.rim_center is None:
            return None
        wx, wy = rel_frame.wrist_xy
        rx, ry = rel_frame.rim_center
        return math.hypot(rx - wx, ry - wy)

    def _calc_arc_peak(self, post_frames) -> float | None:
        """Highest ball position (smallest pixel y) after release."""
        ball_ys = [f.ball_xy[1] for f in post_frames if f.ball_xy is not None]
        if not ball_ys:
            return None
        return min(ball_ys)

    @staticmethod
    def _calc_arc_height_ratio(post_frames, rel_frame: _FrameData | None) -> float | None:
        """
        Ratio of how high the ball rises above the release-to-rim baseline,
        normalised by the horizontal distance of the shot.

        Higher ratio = more arc relative to the shot distance.
        A good mid-range shot is roughly 0.4-0.7; a three is 0.3-0.5.
        """
        if rel_frame is None or rel_frame.wrist_xy is None or rel_frame.rim_center is None:
            return None

        ball_ys = [f.ball_xy[1] for f in post_frames if f.ball_xy is not None]
        if not ball_ys:
            return None

        peak_y = min(ball_ys)  # smallest y = highest on screen

        # Baseline: straight line y from wrist to rim at release
        wy = rel_frame.wrist_xy[1]
        ry = rel_frame.rim_center[1]
        baseline_y = min(wy, ry)  # the higher of the two endpoints

        # How far above the baseline does the ball rise?
        rise = baseline_y - peak_y  # positive = ball went above baseline
        if rise <= 0:
            return 0.0  # ball never went above the baseline

        # Horizontal distance as denominator
        horiz = abs(rel_frame.rim_center[0] - rel_frame.wrist_xy[0])
        if horiz < 1.0:
            return None  # basically under the rim, ratio meaningless

        return round(rise / horiz, 3)

    def _calc_arc_symmetry(self, post_frames, release_frame, result_frame) -> float | None:
        """
        Symmetry of the parabola: ratio of (release→peak time) to
        (peak→result time).  Perfect symmetry = 1.0.
        """
        ball_frames = [(f.frame_idx, f.ball_xy[1])
                       for f in post_frames if f.ball_xy is not None]
        if len(ball_frames) < 3:
            return None

        # Find the frame with the lowest y (highest point on screen)
        peak_idx, _ = min(ball_frames, key=lambda t: t[1])

        t_up   = peak_idx - release_frame
        t_down = result_frame - peak_idx

        if t_up <= 0 or t_down <= 0:
            return None

        ratio = min(t_up, t_down) / max(t_up, t_down)
        return round(ratio, 3)

    def _calc_knee_elbow_lag(self, pre_frames) -> float | None:
        """
        Frames between peak knee extension and peak elbow extension
        in the pre-release window.  Negative = knee extends first (ideal).
        """
        knee_data  = [(f.frame_idx, f.knee_angle)  for f in pre_frames if f.knee_angle is not None]
        elbow_data = [(f.frame_idx, f.elbow_angle) for f in pre_frames if f.elbow_angle is not None]

        if not knee_data or not elbow_data:
            return None

        knee_peak_frame  = max(knee_data,  key=lambda t: t[1])[0]
        elbow_peak_frame = max(elbow_data, key=lambda t: t[1])[0]

        return elbow_peak_frame - knee_peak_frame

    def _calc_shot_tempo(self, pre_frames, release_frame) -> int | None:
        """
        Frames from set point (lowest wrist position in pre-release
        window) to release.
        """
        wrist_data = [(f.frame_idx, f.wrist_y_norm)
                      for f in pre_frames if f.wrist_y_norm is not None]
        if not wrist_data:
            return None

        # Set point = frame where wrist is at its lowest (highest y value)
        set_frame = max(wrist_data, key=lambda t: t[1])[0]

        tempo = release_frame - set_frame
        return max(0, tempo)

    def _calc_torso_drift(self, pre_frames) -> float | None:
        """
        Horizontal pixel movement of shoulder midpoint across the
        pre-release window.  Large values → fading / leaning.
        """
        torso_xs = [f.torso_x for f in pre_frames if f.torso_x is not None]
        if len(torso_xs) < 2:
            return None

        return round(abs(torso_xs[-1] - torso_xs[0]), 1)
