"""
Video processor — runs the full detection pipeline on every frame of a video
file and returns a list of ShotEvent objects (one per confirmed shot attempt).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import cv2
import numpy as np

from detectors import (
    PoseEstimator,
    ReleaseDetector,
    ShotDetector,
    NetMotionDetector,
    ShotMetricsEngine,
    ShotMetrics,
    MistakeEngine,
    Mistake,
    estimate_distance,
)


@dataclass
class ShotEvent:
    """Returned when a shot attempt is confirmed (make or miss)."""
    made: bool
    metrics: ShotMetrics
    mistakes: list[Mistake]
    dist_result: dict | None = None


# ── Ball smoother ────────────────────────────────────────────────────────────

_BALL_SMOOTH_ALPHA = 0.4
_BALL_MAX_JUMP_PX  = 120
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

        if self.miss_count > 0 and self.smooth_xy is not None:
            if self.miss_count <= _BALL_COAST_FRAMES:
                vx, vy = self.velocity
                self.smooth_xy = (self.smooth_xy[0] + vx, self.smooth_xy[1] + vy)

        if self.miss_count > _BALL_EXPIRE_FRAMES:
            self.smooth_xy = None
            self.velocity = (0.0, 0.0)
            self.last_ball_xy = None
        elif self.smooth_xy is not None:
            self.last_ball_xy = (int(self.smooth_xy[0]), int(self.smooth_xy[1]))


# ── VideoProcessor ────────────────────────────────────────────────────────────

class VideoProcessor:
    """
    Processes an entire video file and returns a list of ShotEvent objects.

    Usage::

        proc = VideoProcessor()
        events = proc.process_video("/path/to/clip.mp4")
    """

    WIDTH  = 640
    HEIGHT = 480
    # Process at this effective FPS regardless of source frame rate.
    # Reduces CPU time while preserving enough temporal resolution.
    TARGET_FPS = 15

    def __init__(self) -> None:
        self.pose_estimator = PoseEstimator()
        try:
            self.shot_detector: ShotDetector | None = ShotDetector()
        except FileNotFoundError:
            self.shot_detector = None

        self.release_detector    = ReleaseDetector()
        self.net_motion_detector = NetMotionDetector()
        self.shot_metrics_engine = ShotMetricsEngine()
        self.mistake_engine      = MistakeEngine()
        self._ball               = _BallSmoother()
        self._frame_idx          = 0
        self._release_detected   = False

    # ------------------------------------------------------------------

    def process_video(
        self,
        video_path: str,
        progress_callback=None,  # Optional[Callable[[float], None]]
    ) -> list[ShotEvent]:
        """
        Process all frames and return one ShotEvent per confirmed shot attempt.

        ``progress_callback`` is called with a float 0.0–1.0 after each
        processed frame so callers can update a loading indicator.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        src_fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        skip         = max(1, round(src_fps / self.TARGET_FPS))

        events: list[ShotEvent] = []
        read_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                read_idx += 1
                if (read_idx - 1) % skip != 0:
                    continue

                event = self._process_frame(frame)
                if event is not None:
                    events.append(event)

                if progress_callback is not None:
                    progress_callback(min(read_idx / total_frames, 1.0))
        finally:
            cap.release()

        return events

    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray) -> ShotEvent | None:
        self._frame_idx += 1

        sx = self.WIDTH  / frame.shape[1]
        sy = self.HEIGHT / frame.shape[0]
        frame_resized = cv2.resize(frame, (self.WIDTH, self.HEIGHT))

        # --- YOLO: ball + hoop detection ---
        shot_result = self.shot_detector.update(frame) if self.shot_detector else None

        raw_ball_xy: tuple[int, int] | None = None
        if shot_result and shot_result["ball_bbox"]:
            sx1, sy1, sx2, sy2 = shot_result["ball_bbox"]
            raw_ball_xy = (int((sx1 + sx2) / 2 * sx), int((sy1 + sy2) / 2 * sy))

        self._ball.update(raw_ball_xy)

        # --- Pose estimation ---
        frame_rgb    = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        pose_ball_xy = raw_ball_xy if raw_ball_xy else self._ball.last_ball_xy
        pose_result  = self.pose_estimator.process_frame(frame_rgb, ball_xy=pose_ball_xy)
        landmarks    = pose_result.primary

        if self.pose_estimator.person_switched:
            if hasattr(self.release_detector, "reset"):
                self.release_detector.reset()

        angles: dict | None = None
        if landmarks:
            angles = self.pose_estimator.get_joint_angles(landmarks)

        # --- Hoop bbox from ShotDetector ---
        rim_result: dict | None = None
        if shot_result and shot_result["hoop_bbox"]:
            hx1, hy1, hx2, hy2 = shot_result["hoop_bbox"]
            rim_result = {
                "center": (int((hx1 + hx2) / 2 * sx), int((hy1 + hy2) / 2 * sy)),
                "bbox":   (int(hx1 * sx), int(hy1 * sy), int(hx2 * sx), int(hy2 * sy)),
                "locked": True,
            }

        # --- Metrics feed ---
        rim_center = rim_result["center"] if rim_result else None
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
        if landmarks and angles:
            h, w = frame_resized.shape[:2]
            wrist      = landmarks[16]
            elbow_lm   = landmarks[14]
            shoulder_r = landmarks[12]
            shoulder_l = landmarks[11]

            wx, wy = int(wrist.x * w),      int(wrist.y * h)
            ex, ey = int(elbow_lm.x * w),   int(elbow_lm.y * h)
            slx, sly = int(shoulder_l.x * w), int(shoulder_l.y * h)
            srx, sry = int(shoulder_r.x * w), int(shoulder_r.y * h)

            bx_r = self._ball.last_ball_xy[0] if self._ball.last_ball_xy else wx
            by_r = self._ball.last_ball_xy[1] if self._ball.last_ball_xy else wy

            rel = self.release_detector.detect(
                bx_r, by_r, wx, wy, ex, ey, slx, sly, srx, sry,
                angles["elbow_angle"],
            )
            if rel["release"]:
                self.shot_metrics_engine.on_release(self._frame_idx)
                self._release_detected = True
            else:
                self._release_detected = False

        # --- Net-motion make/miss ---
        hoop_bbox = rim_result["bbox"] if rim_result else None
        armed = shot_result is not None and shot_result.get("state") == "ARMED"
        if not armed and self.shot_detector is None and self._release_detected:
            armed = True

        net_result = self.net_motion_detector.update(frame_resized, hoop_bbox, armed)

        if net_result["attempt"]:
            is_make     = net_result["make"]
            shot_metrics = self.shot_metrics_engine.on_result(self._frame_idx, made=is_make)

            if shot_metrics is None:
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
            return ShotEvent(made=is_make, metrics=shot_metrics, mistakes=mistakes, dist_result=dist_result)

        return None
