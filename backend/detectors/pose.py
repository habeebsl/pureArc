import math
import os
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights", "pose_landmarker_full.task")


# Landmark indices for the MediaPipe 33-point pose model
class PoseLandmark:
    RIGHT_SHOULDER = 12
    RIGHT_ELBOW    = 14
    RIGHT_WRIST    = 16
    RIGHT_HIP      = 24
    RIGHT_KNEE     = 26
    RIGHT_ANKLE    = 28


# Skeleton connections used for drawing
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),   # arms
    (11, 23), (12, 24), (23, 24),                         # torso
    (23, 25), (25, 27), (24, 26), (26, 28),               # legs
    (0, 1),  (1, 2),  (2, 3),  (3, 7),                   # face left
    (0, 4),  (4, 5),  (5, 6),  (6, 8),                   # face right
    (9, 10),                                              # mouth
    (15, 17), (15, 19), (15, 21), (17, 19),               # left hand
    (16, 18), (16, 20), (16, 22), (18, 20),               # right hand
    (27, 29), (27, 31), (29, 31),                         # left foot
    (28, 30), (28, 32), (30, 32),                         # right foot
]


class PoseEstimator:
    # Person-lock: once we identify the ball-handler, track them by torso
    # position so we don't jump between players frame-to-frame.
    _LOCK_RADIUS_PX   = 50     # max deviation from predicted position (px)
    _LOCK_EXPIRE      = 15     # frames before lock resets & re-acquires
    _LOCK_INIT_FRAMES = 3      # consecutive consistent frames to establish lock

    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            print("Downloading pose landmarker model (~30 MB)...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("Model downloaded.")

        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=5,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self._timestamp_ms = 0
        self._locked_torso = None    # (x_px, y_px) of tracked person's torso
        self._lock_vel     = (0.0, 0.0)  # torso velocity (px/frame)
        self._lock_miss    = 0       # consecutive frames without a good match
        self._pending_torso = None   # torso pos being validated before lock
        self._pending_count = 0      # consecutive consistent frames during init
        self.person_switched = False # True when tracked person changes

    def process_frame(self, frame_rgb, ball_xy=None):
        """
        When multiple people are detected the method **locks onto** the
        ball-handler (identified by wrist proximity to *ball_xy*) and
        tracks them by torso position + velocity prediction on subsequent
        frames.

        If MediaPipe flickers to a different person (torso deviates from
        predicted position by > threshold), returns **None** so the release
        detector never receives contaminated data.

        Returns a list of NormalizedLandmark, or None.
        """
        self.person_switched = False
        self._timestamp_ms += 33
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect_for_video(mp_image, self._timestamp_ms)
        self._last_result = result

        if not result.pose_landmarks:
            self._lock_miss += 1
            if self._lock_miss > self._LOCK_EXPIRE:
                self._locked_torso = None
                self._lock_vel = (0.0, 0.0)
                self._pending_torso = None
                self._pending_count = 0
            return None

        h, w = frame_rgb.shape[:2]

        # --- Single person detected ---
        if len(result.pose_landmarks) == 1:
            pose = result.pose_landmarks[0]
            tc = self._torso_center(pose, h, w)

            # Lock not yet established — build up consistency
            if self._locked_torso is None:
                if self._pending_torso is not None:
                    d = math.hypot(tc[0] - self._pending_torso[0],
                                   tc[1] - self._pending_torso[1])
                    if d < self._LOCK_RADIUS_PX:
                        self._pending_count += 1
                        self._pending_torso = tc
                        if self._pending_count >= self._LOCK_INIT_FRAMES:
                            # Consistent enough — establish lock
                            self._commit_lock(tc)
                    else:
                        # Inconsistent — restart
                        self._pending_torso = tc
                        self._pending_count = 0
                else:
                    self._pending_torso = tc
                    self._pending_count = 0
                return pose  # no lock yet, return everything

            # Lock exists — check match via velocity prediction
            if self._lock_dist(tc) > self._LOCK_RADIUS_PX:
                # Wrong person — suppress
                self._lock_miss += 1
                if self._lock_miss > self._LOCK_EXPIRE:
                    # Lock expired — start re-init
                    self._locked_torso = None
                    self._lock_vel = (0.0, 0.0)
                    self._pending_torso = tc
                    self._pending_count = 0
                    self.person_switched = True
                    return pose
                return None

            self._commit_lock(tc)
            return pose

        # --- Multiple people: prefer locked person, fall back to ball ---

        if self._locked_torso is not None:
            closest, dist = self._nearest_to_lock(result.pose_landmarks, h, w)
            if dist < self._LOCK_RADIUS_PX:
                self._commit_lock(self._torso_center(closest, h, w))
                return closest
            # Match too far — don't return wrong person
            self._lock_miss += 1
            if self._lock_miss < self._LOCK_EXPIRE:
                return None
            # Lock expired — fall through to re-acquire via ball
            self._locked_torso = None
            self._lock_vel = (0.0, 0.0)
            self._lock_miss = 0

        # Clear pending — multi-person selection is more reliable
        self._pending_torso = None
        self._pending_count = 0

        # Acquire / re-acquire lock via wrist-ball proximity
        if ball_xy is not None:
            chosen = self._nearest_to_ball(result.pose_landmarks, ball_xy, h, w)
        else:
            chosen = result.pose_landmarks[0]

        self._commit_lock(self._torso_center(chosen, h, w))
        self.person_switched = True
        return chosen

    # ---- internal helpers -------------------------------------------

    def _lock_dist(self, tc):
        """Distance from tc to velocity-predicted lock position."""
        px = self._locked_torso[0] + self._lock_vel[0]
        py = self._locked_torso[1] + self._lock_vel[1]
        return math.hypot(tc[0] - px, tc[1] - py)

    def _commit_lock(self, tc):
        """Update lock position and derive velocity."""
        if self._locked_torso is not None:
            self._lock_vel = (tc[0] - self._locked_torso[0],
                              tc[1] - self._locked_torso[1])
        self._locked_torso = tc
        self._lock_miss = 0

    # ---- internal helpers -------------------------------------------

    @staticmethod
    def _torso_center(pose, h, w):
        """Average of visible shoulder + hip landmarks in pixel coords."""
        xs, ys = [], []
        for idx in (11, 12, 23, 24):  # L/R shoulders, L/R hips
            if pose[idx].visibility > 0.3:
                xs.append(pose[idx].x * w)
                ys.append(pose[idx].y * h)
        if not xs:
            return (w / 2, h / 2)
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def _nearest_to_lock(self, poses, h, w):
        """Return (pose, predicted_distance) closest to velocity-predicted lock."""
        best, best_d = None, float('inf')
        for pose in poses:
            tc = self._torso_center(pose, h, w)
            d = self._lock_dist(tc)
            if d < best_d:
                best_d = d
                best = pose
        return best, best_d

    @staticmethod
    def _nearest_to_ball(poses, ball_xy, h, w):
        """Return the pose whose wrist is closest to ball_xy."""
        bx, by = ball_xy
        best, best_d = None, float('inf')
        for pose in poses:
            for wrist_idx in (16, 15):
                wx = pose[wrist_idx].x * w
                wy = pose[wrist_idx].y * h
                d = math.hypot(bx - wx, by - wy)
                if d < best_d:
                    best_d = d
                    best = pose
        return best

    @staticmethod
    def calculate_angle(a, b, c):
        """
        Calculates angle between three points.
        Points must be (x, y) format.
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (
            np.linalg.norm(ba) * np.linalg.norm(bc)
        )

        # Prevent numerical errors
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

        return np.degrees(np.arccos(cosine_angle))

    def get_joint_angles(self, landmarks):
        """
        Returns elbow and knee angles for right side.
        """
        shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER]
        elbow    = landmarks[PoseLandmark.RIGHT_ELBOW]
        wrist    = landmarks[PoseLandmark.RIGHT_WRIST]

        hip   = landmarks[PoseLandmark.RIGHT_HIP]
        knee  = landmarks[PoseLandmark.RIGHT_KNEE]
        ankle = landmarks[PoseLandmark.RIGHT_ANKLE]

        elbow_angle = self.calculate_angle(
            (shoulder.x, shoulder.y),
            (elbow.x, elbow.y),
            (wrist.x, wrist.y)
        )

        knee_angle = self.calculate_angle(
            (hip.x, hip.y),
            (knee.x, knee.y),
            (ankle.x, ankle.y)
        )

        return {
            "elbow_angle": elbow_angle,
            "knee_angle": knee_angle,
        }