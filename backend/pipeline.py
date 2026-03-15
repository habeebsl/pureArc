"""
Reusable per-session CV detection pipeline.

Encapsulates all detectors from main.py (pose, release, shot, net-motion,
metrics, mistakes) behind a single `process_frame(frame) -> ShotEvent | None`
interface so both the standalone main.py loop and the FastAPI frame-ingest
endpoint can drive shot detection.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from detectors import (
    PoseEstimator,
    ReleaseDetector,
    RimDetector,
    ShotDetector,
    NetMotionDetector,
    ShotMetricsEngine,
    ShotMetrics,
    MistakeEngine,
    Mistake,
    estimate_distance,
)


_CUSTOM_WEIGHTS = os.path.join(
    os.path.dirname(__file__), "runs", "rim_detector", "weights", "best.pt"
)


@dataclass
class ShotEvent:
    """Returned when a shot attempt is confirmed (make or miss)."""

    made: bool
    metrics: ShotMetrics
    mistakes: list[Mistake]
    dist_result: dict | None = None


# ── Ball smoother (stateful, per-pipeline instance) ────────────────────

_BALL_SMOOTH_ALPHA = 0.4
_BALL_MAX_JUMP_PX = 120
_BALL_COAST_FRAMES = 8
_BALL_EXPIRE_FRAMES = 15


class _BallSmoother:
    """EMA + outlier rejection + coasting for raw YOLO ball detections."""

    def __init__(self):
        self.smooth_xy: tuple[float, float] | None = None
        self.velocity: tuple[float, float] = (0.0, 0.0)
        self.miss_count: int = 0
        self.last_ball_xy: tuple[int, int] | None = None

    def update(self, raw_xy: tuple[int, int] | None) -> None:
        if raw_xy is not None:
            rx, ry = float(raw_xy[0]), float(raw_xy[1])

            if self.smooth_xy is None:
                self.smooth_xy = (rx, ry)
                self.velocity = (0.0, 0.0)
                self.miss_count = 0
            else:
                dx = rx - self.smooth_xy[0]
                dy = ry - self.smooth_xy[1]
                dist = math.hypot(dx, dy)

                if dist <= _BALL_MAX_JUMP_PX:
                    a = _BALL_SMOOTH_ALPHA
                    sx = self.smooth_xy[0] * (1 - a) + rx * a
                    sy = self.smooth_xy[1] * (1 - a) + ry * a
                    self.velocity = (sx - self.smooth_xy[0], sy - self.smooth_xy[1])
                    self.smooth_xy = (sx, sy)
                    self.miss_count = 0
                else:
                    if self.miss_count >= _BALL_COAST_FRAMES:
                        self.smooth_xy = (rx, ry)
                        self.velocity = (0.0, 0.0)
                        self.miss_count = 0
                    else:
                        self.miss_count += 1
        else:
            self.miss_count += 1

        # Coast
        if self.miss_count > 0 and self.smooth_xy is not None:
            if self.miss_count <= _BALL_COAST_FRAMES:
                vx, vy = self.velocity
                self.smooth_xy = (self.smooth_xy[0] + vx, self.smooth_xy[1] + vy)

        # Expire
        if self.miss_count > _BALL_EXPIRE_FRAMES:
            self.smooth_xy = None
            self.velocity = (0.0, 0.0)
            self.last_ball_xy = None
        elif self.smooth_xy is not None:
            self.last_ball_xy = (int(self.smooth_xy[0]), int(self.smooth_xy[1]))


# ── Pipeline ───────────────────────────────────────────────────────────

# Shared heavyweight models (loaded once, shared across all Pipeline instances)
_shared_pose: PoseEstimator | None = None
_shared_shot: ShotDetector | None = None
_shared_rim: RimDetector | None = None


def _get_shared_models() -> tuple[PoseEstimator, ShotDetector | None, RimDetector]:
    global _shared_pose, _shared_shot, _shared_rim

    if _shared_pose is None:
        _shared_pose = PoseEstimator()

    if _shared_shot is None:
        try:
            _shared_shot = ShotDetector()
        except FileNotFoundError:
            _shared_shot = None  # type: ignore[assignment]

    if _shared_rim is None:
        _shared_rim = RimDetector(
            custom_model_path=_CUSTOM_WEIGHTS if os.path.exists(_CUSTOM_WEIGHTS) else None
        )

    return _shared_pose, _shared_shot, _shared_rim


class Pipeline:
    """Per-session CV detection pipeline.

    Usage:
        pipe = Pipeline()
        for frame in frames:
            event = pipe.process_frame(frame)
            if event is not None:
                # a shot was detected
    """

    # Target processing size
    WIDTH = 640
    HEIGHT = 480

    def __init__(self) -> None:
        self.pose_estimator, self.shot_detector, self.rim_detector = _get_shared_models()
        self.release_detector = ReleaseDetector()
        self.net_motion_detector = NetMotionDetector()
        self.shot_metrics_engine = ShotMetricsEngine()
        self.mistake_engine = MistakeEngine()
        self._ball = _BallSmoother()
        self._frame_idx = 0

    def process_frame(self, frame: np.ndarray) -> ShotEvent | None:
        """Process a single BGR frame. Returns ShotEvent if a shot was confirmed."""
        self._frame_idx += 1

        # Keep native frame for shot detector (YOLO handles its own resize)
        shot_frame = frame
        sx = self.WIDTH / frame.shape[1]
        sy = self.HEIGHT / frame.shape[0]

        # Resize for the rest of the pipeline
        frame_resized = cv2.resize(frame, (self.WIDTH, self.HEIGHT))

        # --- YOLO: ball + hoop detection ---
        shot_result = self.shot_detector.update(shot_frame) if self.shot_detector else None

        raw_ball_xy: tuple[int, int] | None = None
        if shot_result and shot_result["ball_bbox"]:
            sx1, sy1, sx2, sy2 = shot_result["ball_bbox"]
            bx = int((sx1 + sx2) / 2 * sx)
            by = int((sy1 + sy2) / 2 * sy)
            raw_ball_xy = (bx, by)

        self._ball.update(raw_ball_xy)

        # --- Pose estimation ---
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        pose_ball_xy = raw_ball_xy if raw_ball_xy else self._ball.last_ball_xy
        pose_result = self.pose_estimator.process_frame(frame_rgb, ball_xy=pose_ball_xy)
        landmarks = pose_result.primary

        if self.pose_estimator.person_switched:
            self.release_detector.reset()

        angles: dict | None = None
        if landmarks:
            angles = self.pose_estimator.get_joint_angles(landmarks)

        # --- Hoop / rim ---
        if shot_result and shot_result["hoop_bbox"]:
            hx1, hy1, hx2, hy2 = shot_result["hoop_bbox"]
            hx1s, hy1s = int(hx1 * sx), int(hy1 * sy)
            hx2s, hy2s = int(hx2 * sx), int(hy2 * sy)
            rim_result: dict | None = {
                "center": ((hx1s + hx2s) // 2, (hy1s + hy2s) // 2),
                "bbox": (hx1s, hy1s, hx2s, hy2s),
                "locked": True,
            }
        else:
            rim_result = self.rim_detector.detect_rim(frame_resized)

        # --- Metrics feed ---
        rim_center = rim_result["center"] if rim_result and "center" in rim_result else None
        self.shot_metrics_engine.feed(
            self._frame_idx,
            landmarks,
            self._ball.last_ball_xy,
            rim_center,
            angles,
            frame_hw=(self.HEIGHT, self.WIDTH),
        )

        # --- Distance estimation ---
        dist_result = None
        if rim_result and landmarks:
            dist_result = estimate_distance(rim_result, landmarks, frame_resized.shape)

        # --- Release detection ---
        if landmarks:
            h, w = frame_resized.shape[:2]
            wrist = landmarks[16]
            elbow_lm = landmarks[14]
            shoulder_r = landmarks[12]
            shoulder_l = landmarks[11]

            wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
            elbow_x, elbow_y = int(elbow_lm.x * w), int(elbow_lm.y * h)
            sl_x, sl_y = int(shoulder_l.x * w), int(shoulder_l.y * h)
            sr_x, sr_y = int(shoulder_r.x * w), int(shoulder_r.y * h)

            bx_r = self._ball.last_ball_xy[0] if self._ball.last_ball_xy else wrist_x
            by_r = self._ball.last_ball_xy[1] if self._ball.last_ball_xy else wrist_y

            result = self.release_detector.detect(
                bx_r, by_r,
                wrist_x, wrist_y,
                elbow_x, elbow_y,
                sl_x, sl_y,
                sr_x, sr_y,
                angles["elbow_angle"],
            )
            if result["release"]:
                self.shot_metrics_engine.on_release(self._frame_idx)

        # --- Net-motion make/miss ---
        hoop_bbox = rim_result["bbox"] if rim_result and "bbox" in rim_result else None
        armed = shot_result is not None and shot_result["state"] == "ARMED"
        net_result = self.net_motion_detector.update(frame_resized, hoop_bbox, armed)

        if net_result["attempt"]:
            is_make = net_result["make"]
            shot_metrics = self.shot_metrics_engine.on_result(self._frame_idx, made=is_make)

            if shot_metrics is None:
                # No pending release — create minimal metrics
                shot_metrics = ShotMetrics(
                    release_angle=None,
                    release_height=None,
                    elbow_angle=angles["elbow_angle"] if angles else None,
                    arc_peak=None,
                    arc_height_ratio=None,
                    arc_symmetry=None,
                    knee_elbow_lag=None,
                    shot_tempo=None,
                    shot_distance_px=None,
                    torso_drift=None,
                    made=is_make,
                    release_frame=None,
                    result_frame=self._frame_idx,
                )

            mistakes = self.mistake_engine.analyse(shot_metrics)

            return ShotEvent(
                made=is_make,
                metrics=shot_metrics,
                mistakes=mistakes,
                dist_result=dist_result,
            )

        return None
