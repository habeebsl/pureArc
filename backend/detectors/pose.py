"""
RTMPose estimator — multi-player aware.

Uses rtmlib (lightweight ONNX-based RTMPose inference).  Landmarks are
returned as a 33-element list whose indices match MediaPipe's numbering
so that main.py, distance.py, and release.py need zero changes.

COCO keypoints (17) are mapped into the MediaPipe slots; unmapped slots
(hands, face detail, heels, foot-index) are filled with visibility = 0.

process_frame() returns a PoseResult containing:
  • all_poses   — every detected person (normalised 0-1)
  • primary     — the tracked/selected player's landmarks (or None)
  • primary_idx — index into all_poses (or -1)

To switch the tracked player call select_primary(pixel_xy) with the
tap location — the person whose torso is closest will become primary.
"""

import math
from dataclasses import dataclass, field

import numpy as np
from rtmlib import Body


# ── COCO-to-MediaPipe index mapping ────────────────────────────────────
_COCO_TO_MP = {
    0:  0,   # nose
    1:  2,   # left eye
    2:  5,   # right eye
    3:  7,   # left ear
    4:  8,   # right ear
    5:  11,  # left shoulder
    6:  12,  # right shoulder
    7:  13,  # left elbow
    8:  14,  # right elbow
    9:  15,  # left wrist
    10: 16,  # right wrist
    11: 23,  # left hip
    12: 24,  # right hip
    13: 25,  # left knee
    14: 26,  # right knee
    15: 27,  # left ankle
    16: 28,  # right ankle
}

_MP_SLOTS = 33  # total MediaPipe landmark count


# ── Lightweight landmark object ────────────────────────────────────────
class _Landmark:
    """Mimics mediapipe NormalizedLandmark."""
    __slots__ = ("x", "y", "visibility")

    def __init__(self, x: float = 0.0, y: float = 0.0, visibility: float = 0.0):
        self.x = x
        self.y = y
        self.visibility = visibility


def _empty_landmarks():
    return [_Landmark() for _ in range(_MP_SLOTS)]


def _coco_to_landmarks(kpt_xy, kpt_scores):
    """
    Convert a single person's COCO keypoints to 33-element MediaPipe list.

    Parameters
    ----------
    kpt_xy     : ndarray (17, 2)  — x_px, y_px
    kpt_scores : ndarray (17,)    — confidence
    """
    lms = _empty_landmarks()
    for coco_idx, mp_idx in _COCO_TO_MP.items():
        x_px, y_px = kpt_xy[coco_idx]
        conf = float(kpt_scores[coco_idx])
        lms[mp_idx] = _Landmark(x_px, y_px, conf)
    return lms


# ── Skeleton connections (MediaPipe indices, same subset COCO covers) ──
POSE_CONNECTIONS = [
    (11, 12),                          # shoulders
    (11, 13), (13, 15),                # left arm
    (12, 14), (14, 16),                # right arm
    (11, 23), (12, 24), (23, 24),      # torso
    (23, 25), (25, 27),                # left leg
    (24, 26), (26, 28),                # right leg
]


# ── MediaPipe-compatible landmark index constants ──────────────────────
class PoseLandmark:
    RIGHT_SHOULDER = 12
    RIGHT_ELBOW    = 14
    RIGHT_WRIST    = 16
    RIGHT_HIP      = 24
    RIGHT_KNEE     = 26
    RIGHT_ANKLE    = 28


@dataclass
class PoseResult:
    """Result of a single frame's pose estimation."""
    all_poses:   list        # list of landmark lists (one per person, normalised 0-1)
    primary:     object      # landmarks for the tracked player, or None
    primary_idx: int = -1    # index into all_poses (-1 = none)


# ── Main estimator ────────────────────────────────────────────────────
class PoseEstimator:
    _LOCK_RADIUS_PX   = 50
    _LOCK_EXPIRE      = 15
    _LOCK_INIT_FRAMES = 3
    _EMA_ALPHA        = 0.5  # keypoint smoothing (0 = no update, 1 = no smooth)

    def __init__(self, mode: str = "balanced"):
        """
        Parameters
        ----------
        mode : 'lightweight' | 'balanced' | 'performance'
            Controls model size / accuracy trade-off.
        """
        self.body = Body(mode=mode, backend="onnxruntime", device="cpu")

        self._locked_torso  = None
        self._lock_vel      = (0.0, 0.0)
        self._lock_miss     = 0
        self._pending_torso = None
        self._pending_count = 0
        self.person_switched = False
        # Per-person smooth buffers keyed by person index
        self._smooth_bufs: dict[int, list] = {}
        self._primary_smooth_key: int | None = None
        # Manual selection: set via select_primary()
        self._manual_lock_xy: tuple | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def process_frame(self, frame_rgb, ball_xy=None) -> PoseResult:
        """
        Run RTMPose on *frame_rgb* (RGB uint8 H×W×3).

        Returns PoseResult with all detected poses and the primary player.
        """
        self.person_switched = False
        h, w = frame_rgb.shape[:2]

        # rtmlib expects BGR
        frame_bgr = frame_rgb[:, :, ::-1]
        keypoints, scores = self.body(frame_bgr)

        if keypoints is None or len(keypoints) == 0:
            self._lock_miss += 1
            if self._lock_miss > self._LOCK_EXPIRE:
                self._reset_lock()
            return PoseResult(all_poses=[], primary=None, primary_idx=-1)

        n_people = keypoints.shape[0]

        # Build per-person landmark lists (pixel coords)
        raw_poses = []
        for i in range(n_people):
            lms = _coco_to_landmarks(keypoints[i], scores[i])
            raw_poses.append(lms)

        # ── Select primary player ─────────────────────────────────── #
        primary_raw, primary_raw_idx = self._select_primary(raw_poses, ball_xy, h, w)

        # ── Finalise all poses (normalise to 0-1) ────────────────── #
        all_normalised = []
        for i, pose in enumerate(raw_poses):
            is_primary = (i == primary_raw_idx)
            norm = self._finalize(pose, h, w, person_key=i, smooth=is_primary)
            all_normalised.append(norm)

        primary_norm = all_normalised[primary_raw_idx] if primary_raw_idx >= 0 else None

        return PoseResult(
            all_poses=all_normalised,
            primary=primary_norm,
            primary_idx=primary_raw_idx,
        )

    def select_primary(self, pixel_xy: tuple, frame_hw: tuple = (480, 640)):
        """
        Tap-to-select: lock onto the person whose torso is closest to
        *pixel_xy* (x, y in pixel coords).  Takes effect on the next
        process_frame() call.
        """
        self._manual_lock_xy = pixel_xy
        # Force re-acquisition
        self._locked_torso = None
        self._lock_vel = (0.0, 0.0)
        self._lock_miss = 0
        self._pending_torso = None
        self._pending_count = 0

    def get_joint_angles(self, landmarks):
        shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER]
        elbow    = landmarks[PoseLandmark.RIGHT_ELBOW]
        wrist    = landmarks[PoseLandmark.RIGHT_WRIST]
        hip      = landmarks[PoseLandmark.RIGHT_HIP]
        knee     = landmarks[PoseLandmark.RIGHT_KNEE]
        ankle    = landmarks[PoseLandmark.RIGHT_ANKLE]

        elbow_angle = self.calculate_angle(
            (shoulder.x, shoulder.y),
            (elbow.x, elbow.y),
            (wrist.x, wrist.y),
        )
        knee_angle = self.calculate_angle(
            (hip.x, hip.y),
            (knee.x, knee.y),
            (ankle.x, ankle.y),
        )
        return {"elbow_angle": elbow_angle, "knee_angle": knee_angle}

    @staticmethod
    def calculate_angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
        cos = np.clip(cos, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos)))

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _select_primary(self, raw_poses, ball_xy, h, w):
        """
        Pick the primary player from raw_poses (pixel coords).
        Returns (pose, index) or (None, -1).
        """
        n = len(raw_poses)

        # ── Manual selection via tap ──
        if self._manual_lock_xy is not None:
            mx, my = self._manual_lock_xy
            best_i, best_d = -1, float("inf")
            for i, pose in enumerate(raw_poses):
                tc = self._torso_center(pose, h, w)
                d = math.hypot(tc[0] - mx, tc[1] - my)
                if d < best_d:
                    best_d = d
                    best_i = i
            if best_i >= 0:
                tc = self._torso_center(raw_poses[best_i], h, w)
                self._commit_lock(tc)
                self._manual_lock_xy = None
                self.person_switched = True
                return raw_poses[best_i], best_i
            self._manual_lock_xy = None

        # ── Single person ──
        if n == 1:
            pose = raw_poses[0]
            tc = self._torso_center(pose, h, w)

            if self._locked_torso is None:
                if self._pending_torso is not None:
                    d = math.hypot(tc[0] - self._pending_torso[0],
                                   tc[1] - self._pending_torso[1])
                    if d < self._LOCK_RADIUS_PX:
                        self._pending_count += 1
                        self._pending_torso = tc
                        if self._pending_count >= self._LOCK_INIT_FRAMES:
                            self._commit_lock(tc)
                    else:
                        self._pending_torso = tc
                        self._pending_count = 0
                else:
                    self._pending_torso = tc
                    self._pending_count = 0
                return pose, 0

            if self._lock_dist(tc) > self._LOCK_RADIUS_PX:
                self._lock_miss += 1
                if self._lock_miss > self._LOCK_EXPIRE:
                    self._reset_lock()
                    self._pending_torso = tc
                    self._pending_count = 0
                    self.person_switched = True
                    return pose, 0
                return None, -1

            self._commit_lock(tc)
            return pose, 0

        # ── Multiple people: prefer locked person, fall back to ball ──

        if self._locked_torso is not None:
            closest_i, dist = self._nearest_to_lock_idx(raw_poses, h, w)
            if dist < self._LOCK_RADIUS_PX:
                self._commit_lock(self._torso_center(raw_poses[closest_i], h, w))
                return raw_poses[closest_i], closest_i
            self._lock_miss += 1
            if self._lock_miss < self._LOCK_EXPIRE:
                return None, -1
            self._reset_lock()

        self._pending_torso = None
        self._pending_count = 0

        if ball_xy is not None:
            chosen_i = self._nearest_to_ball_idx(raw_poses, ball_xy, h, w)
        else:
            chosen_i = 0

        self._commit_lock(self._torso_center(raw_poses[chosen_i], h, w))
        self.person_switched = True
        return raw_poses[chosen_i], chosen_i

    def _finalize(self, landmarks, h, w, person_key=0, smooth=True):
        """Optionally smooth keypoints via EMA then normalise to 0-1."""
        if smooth:
            self._smooth(landmarks, person_key)
        for lm in landmarks:
            if lm.visibility > 0:
                lm.x /= w
                lm.y /= h
        return landmarks

    def _smooth(self, landmarks, person_key=0):
        """Apply EMA smoothing to visible landmark positions (pixel coords)."""
        a = self._EMA_ALPHA
        buf = self._smooth_bufs.get(person_key)
        if buf is None:
            self._smooth_bufs[person_key] = [
                (lm.x, lm.y, lm.visibility) for lm in landmarks
            ]
            return
        for i, lm in enumerate(landmarks):
            px, py, pv = buf[i]
            if lm.visibility > 0.3 and pv > 0.3:
                lm.x = px * (1 - a) + lm.x * a
                lm.y = py * (1 - a) + lm.y * a
                buf[i] = (lm.x, lm.y, lm.visibility)
            elif lm.visibility > 0.3:
                buf[i] = (lm.x, lm.y, lm.visibility)
            else:
                buf[i] = (px, py, pv * 0.8)

    def _reset_lock(self):
        self._locked_torso  = None
        self._lock_vel      = (0.0, 0.0)
        self._lock_miss     = 0
        self._pending_torso = None
        self._pending_count = 0
        self._smooth_bufs.clear()

    def _lock_dist(self, tc):
        px = self._locked_torso[0] + self._lock_vel[0]
        py = self._locked_torso[1] + self._lock_vel[1]
        return math.hypot(tc[0] - px, tc[1] - py)

    def _commit_lock(self, tc):
        if self._locked_torso is not None:
            self._lock_vel = (tc[0] - self._locked_torso[0],
                              tc[1] - self._locked_torso[1])
        self._locked_torso = tc
        self._lock_miss = 0

    @staticmethod
    def _torso_center(pose, h, w):
        """Torso centre in pixel coords (landmarks are still in px here)."""
        xs, ys = [], []
        for idx in (11, 12, 23, 24):
            lm = pose[idx]
            if lm.visibility > 0.3:
                xs.append(lm.x)
                ys.append(lm.y)
        if not xs:
            return (w / 2, h / 2)
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def _nearest_to_lock_idx(self, poses, h, w):
        """Return (index, distance) of the pose closest to velocity-predicted lock."""
        best_i, best_d = -1, float("inf")
        for i, pose in enumerate(poses):
            tc = self._torso_center(pose, h, w)
            d = self._lock_dist(tc)
            if d < best_d:
                best_d = d
                best_i = i
        return best_i, best_d

    @staticmethod
    def _nearest_to_ball_idx(poses, ball_xy, h, w):
        """Return index of the pose whose wrist is closest to ball_xy."""
        bx, by = ball_xy
        best_i, best_d = 0, float("inf")
        for i, pose in enumerate(poses):
            for wrist_idx in (16, 15):
                lm = pose[wrist_idx]
                d = math.hypot(bx - lm.x, by - lm.y)
                if d < best_d:
                    best_d = d
                    best_i = i
        return best_i
