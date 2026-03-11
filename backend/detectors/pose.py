"""
RTMPose estimator — drop-in replacement for the MediaPipe-based
PoseEstimator in pose.py.

Uses rtmlib (lightweight ONNX-based RTMPose inference).  Landmarks are
returned as a 33-element list whose indices match MediaPipe's numbering
so that main.py, distance.py, and release.py need zero changes.

COCO keypoints (17) are mapped into the MediaPipe slots; unmapped slots
(hands, face detail, heels, foot-index) are filled with visibility = 0.
"""

import math

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
        self._smooth_buf    = None  # list of (x, y, vis) per 33 slots

    # ------------------------------------------------------------------ #
    # Public API (matches MediaPipe PoseEstimator)
    # ------------------------------------------------------------------ #

    def process_frame(self, frame_rgb, ball_xy=None):
        """
        Run RTMPose on *frame_rgb* (RGB uint8 H×W×3).

        Returns a list of 33 NormalizedLandmark-like objects, or None.
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
            return None

        n_people = keypoints.shape[0]

        # Build per-person landmark lists (pixel coords initially)
        poses = []
        for i in range(n_people):
            lms = _coco_to_landmarks(keypoints[i], scores[i])
            poses.append(lms)

        # ── Person selection logic (same as MediaPipe version) ──

        if n_people == 1:
            chosen = poses[0]
            tc = self._torso_center(chosen, h, w)

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
                return self._finalize(chosen, h, w)

            if self._lock_dist(tc) > self._LOCK_RADIUS_PX:
                self._lock_miss += 1
                if self._lock_miss > self._LOCK_EXPIRE:
                    self._reset_lock()
                    self._pending_torso = tc
                    self._pending_count = 0
                    self.person_switched = True
                    return self._finalize(chosen, h, w)
                return None

            self._commit_lock(tc)
            return self._finalize(chosen, h, w)

        # ── Multiple people ──

        if self._locked_torso is not None:
            closest, dist = self._nearest_to_lock(poses, h, w)
            if dist < self._LOCK_RADIUS_PX:
                self._commit_lock(self._torso_center(closest, h, w))
                return self._finalize(closest, h, w)
            self._lock_miss += 1
            if self._lock_miss < self._LOCK_EXPIRE:
                return None
            self._reset_lock()

        self._pending_torso = None
        self._pending_count = 0

        if ball_xy is not None:
            chosen = self._nearest_to_ball(poses, ball_xy, h, w)
        else:
            chosen = poses[0]

        self._commit_lock(self._torso_center(chosen, h, w))
        self.person_switched = True
        return self._finalize(chosen, h, w)

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

    def _finalize(self, landmarks, h, w):
        """Smooth keypoints via EMA then normalise to 0-1."""
        self._smooth(landmarks)
        for lm in landmarks:
            if lm.visibility > 0:
                lm.x /= w
                lm.y /= h
        return landmarks

    def _smooth(self, landmarks):
        """Apply EMA smoothing to visible landmark positions (pixel coords)."""
        a = self._EMA_ALPHA
        if self._smooth_buf is None:
            self._smooth_buf = [
                (lm.x, lm.y, lm.visibility) for lm in landmarks
            ]
            return
        for i, lm in enumerate(landmarks):
            px, py, pv = self._smooth_buf[i]
            if lm.visibility > 0.3 and pv > 0.3:
                lm.x = px * (1 - a) + lm.x * a
                lm.y = py * (1 - a) + lm.y * a
                self._smooth_buf[i] = (lm.x, lm.y, lm.visibility)
            elif lm.visibility > 0.3:
                self._smooth_buf[i] = (lm.x, lm.y, lm.visibility)
            else:
                self._smooth_buf[i] = (px, py, pv * 0.8)

    def _reset_lock(self):
        self._locked_torso  = None
        self._lock_vel      = (0.0, 0.0)
        self._lock_miss     = 0
        self._pending_torso = None
        self._pending_count = 0
        self._smooth_buf    = None

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

    def _nearest_to_lock(self, poses, h, w):
        best, best_d = None, float("inf")
        for pose in poses:
            tc = self._torso_center(pose, h, w)
            d = self._lock_dist(tc)
            if d < best_d:
                best_d = d
                best = pose
        return best, best_d

    @staticmethod
    def _nearest_to_ball(poses, ball_xy, h, w):
        bx, by = ball_xy
        best, best_d = None, float("inf")
        for pose in poses:
            for wrist_idx in (16, 15):
                lm = pose[wrist_idx]
                d = math.hypot(bx - lm.x, by - lm.y)
                if d < best_d:
                    best_d = d
                    best = pose
        return best
