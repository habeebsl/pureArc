import cv2
import os
import math
from pose import PoseEstimator, POSE_CONNECTIONS
from release import ReleaseDetector
from ball import BallDetector
from rim import RimDetector
from distance import estimate_distance, draw_distance_overlay

# Use the custom-trained rim detector if weights exist, else fall back to YOLOWorld+HSV
_CUSTOM_WEIGHTS = os.path.join(os.path.dirname(__file__), "runs", "rim_detector", "weights", "best.pt")


def draw_landmarks(frame, landmarks):
    """Draw pose skeleton on frame using OpenCV."""
    h, w = frame.shape[:2]

    # Draw connections
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start = landmarks[start_idx]
            end   = landmarks[end_idx]
            if start.visibility > 0.5 and end.visibility > 0.5:
                pt1 = (int(start.x * w), int(start.y * h))
                pt2 = (int(end.x * w),   int(end.y * h))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    # Draw landmark points
    for lm in landmarks:
        if lm.visibility > 0.5:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)


# Initialize pose estimator (downloads model on first run)
pose_estimator = PoseEstimator()
release_detector = ReleaseDetector()
ball_detector = BallDetector()
rim_detector = RimDetector(
    custom_model_path=_CUSTOM_WEIGHTS if os.path.exists(_CUSTOM_WEIGHTS) else None
)

# Load video (replace with your file)
cap = cv2.VideoCapture("test_shot4.mp4")

# Set up video writer for headless output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output.mp4", fourcc, 30.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for performance (optional but recommended)
    frame = cv2.resize(frame, (640, 480))

    # Convert BGR → RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    landmarks = pose_estimator.process_frame(frame_rgb)

    if landmarks:
        angles = pose_estimator.get_joint_angles(landmarks)

        print(
            f"Elbow: {angles['elbow_angle']:.2f} | "
            f"Knee: {angles['knee_angle']:.2f}"
        )

        elbow = angles['elbow_angle']

        # Draw skeleton
        draw_landmarks(frame, landmarks)

    # --- Rim detection (stationary; locks after a few consistent frames) ---
    rim_result = rim_detector.detect_rim(frame)

    ball_result = ball_detector.detect_ball(frame)

    # Always draw ball detection result so we can visually verify it independently
    if ball_result:
        bx, by, br = ball_result
        cv2.circle(frame, (bx, by), br,  (255, 0, 0), 2)   # ring sized to detected radius
        cv2.circle(frame, (bx, by), 3,   (255, 0, 0), -1)  # center dot
        cv2.putText(frame, f"ball r={br}", (bx + br + 4, by),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        print(f"Ball detected at ({bx}, {by}) r={br}")
    else:
        print("Ball: not detected")

    # --- Distance estimation (rim + pose) ---
    dist_result = None
    if rim_result and landmarks:
        dist_result = estimate_distance(rim_result, landmarks, frame.shape)
        draw_distance_overlay(frame, dist_result, rim_result)
        if dist_result:
            print(
                f"Distance to rim: {dist_result['distance_m']:.2f} m  "
                f"({dist_result['distance_ft']:.1f} ft)  "
                f"[{dist_result['method']}]"
            )
    elif rim_result:
        draw_distance_overlay(frame, None, rim_result)

    if ball_result and landmarks:
        h, w = frame.shape[:2]

        # Right-side landmarks (shooting hand)
        wrist      = landmarks[16]
        elbow_lm   = landmarks[14]
        shoulder_r = landmarks[12]
        shoulder_l = landmarks[11]

        wrist_x = int(wrist.x * w)
        wrist_y = int(wrist.y * h)
        elbow_x = int(elbow_lm.x * w)
        elbow_y = int(elbow_lm.y * h)
        sl_x    = int(shoulder_l.x * w)
        sl_y    = int(shoulder_l.y * h)
        sr_x    = int(shoulder_r.x * w)
        sr_y    = int(shoulder_r.y * h)

        bx, by, br = ball_result
        elbow_angle = angles['elbow_angle']

        # Draw wrist
        cv2.circle(frame, (wrist_x, wrist_y), 6, (0, 255, 255), -1)

        result = release_detector.detect(
            bx, by,
            wrist_x, wrist_y,
            elbow_x, elbow_y,
            sl_x, sl_y,
            sr_x, sr_y,
            elbow_angle,
        )

        # Debug overlay — state + confidence
        conf_pct = int(result['confidence'] * 100)
        state_str = result['state']
        cv2.putText(frame, f"{state_str}  {conf_pct}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

        if result['release']:
            print(f"RELEASE DETECTED  confidence={result['confidence']:.2f}  signals={result['signals']}")
            cv2.putText(frame, "RELEASE", (50, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.4, (0, 0, 255), 3)

    out.write(frame)

cap.release()
out.release()
print("Output saved to output.mp4")