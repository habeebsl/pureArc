import cv2
import math
from pose import PoseEstimator, POSE_CONNECTIONS
from release import ReleaseDetector
from ball import BallDetector


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

# Load video (replace with your file)
cap = cv2.VideoCapture("test_shot.mp4")

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

    if ball_result and landmarks:
        h, w = frame.shape[:2]

        wrist = landmarks[16]  # right wrist
        wrist_x = int(wrist.x * w)
        wrist_y = int(wrist.y * h)

        bx, by, br = ball_result
        elbow_angle = angles['elbow_angle']

        # Draw wrist
        cv2.circle(frame, (wrist_x, wrist_y), 6, (0, 255, 255), -1)

        is_release = release_detector.detect(
            wrist_x, wrist_y,
            bx, by,
            elbow_angle
        )

        if is_release:
            print("🔥 RELEASE DETECTED")
            cv2.putText(frame, "RELEASE", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 255), 3)

    cv2.imshow("PureArc Pose Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()