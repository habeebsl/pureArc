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
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_full.task")


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
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            print("Downloading pose landmarker model (~30 MB)...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("Model downloaded.")

        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def process_frame(self, frame_rgb):
        """
        Expects frame in RGB format (numpy uint8 array).
        Returns a list of NormalizedLandmark if a pose is detected, else None.
        """
        self._timestamp_ms += 33  # ~30 fps increment
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect_for_video(mp_image, self._timestamp_ms)
        self._last_result = result
        if result.pose_landmarks:
            return result.pose_landmarks[0]  # first detected person
        return None

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