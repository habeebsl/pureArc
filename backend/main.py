import cv2
import os
import math
import subprocess
import tempfile
import numpy as np
from detectors import (
    PoseEstimator, POSE_CONNECTIONS,
    ReleaseDetector,
    BallDetector,
    RimDetector,
    estimate_distance, draw_distance_overlay,
    ShotDetector,
)

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
pose_estimator    = PoseEstimator()
release_detector  = ReleaseDetector()
ball_detector     = BallDetector()
rim_detector      = RimDetector(
    custom_model_path=_CUSTOM_WEIGHTS if os.path.exists(_CUSTOM_WEIGHTS) else None
)

# ShotDetector (trajectory-based) — requires trained weights; skipped gracefully if absent
try:
    shot_detector = ShotDetector()
    print("ShotDetector loaded (trajectory scoring active)")
except FileNotFoundError as _e:
    shot_detector = None
    print(f"ShotDetector unavailable: {_e}")

# Score display persistence — keep "SCORE!" on screen for this many frames
_SCORE_DISPLAY_FRAMES = 45
_score_display_cnt    = 0
_score_total          = 0

# Make / Miss flash overlay
_FADE_FRAMES    = 20
_fade_counter   = 0
_overlay_color  = (0, 0, 0)
_overlay_text   = ""

# Release text persistence
_RELEASE_DISPLAY_FRAMES = 30
_release_display_cnt    = 0

# Load video (replace with your file)
cap = cv2.VideoCapture("test_shot4.mp4")

# Set up video writer — write raw mp4v to a temp file, re-encode to H.264 at the end
_OUTPUT_FILE = "output.mp4"
_TEMP_FILE   = "output_tmp.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(_TEMP_FILE, fourcc, 30.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Keep native frame for shot detector — YOLO handles its own resize internally
    # and distorting portrait videos to 640×480 breaks coordinate geometry.
    _shot_frame = frame
    _shot_sx    = 640 / frame.shape[1]
    _shot_sy    = 480 / frame.shape[0]

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
            _release_display_cnt = _RELEASE_DISPLAY_FRAMES

    # Persist RELEASE text for 30 frames
    if _release_display_cnt > 0:
        _release_display_cnt -= 1
        cv2.putText(frame, "RELEASE", (50, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)

    # --- Trajectory-based scoring (shot detector) ---
    shot_result = shot_detector.update(_shot_frame) if shot_detector is not None else None

    # Draw shot-detector detections for visual debug (scale from native → 640×480)
    if shot_result is not None:
        if shot_result["ball_bbox"]:
            sx1, sy1, sx2, sy2 = shot_result["ball_bbox"]
            sx1,sy1,sx2,sy2 = int(sx1*_shot_sx),int(sy1*_shot_sy),int(sx2*_shot_sx),int(sy2*_shot_sy)
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (255, 128, 0), 1)
        if shot_result["hoop_bbox"]:
            hx1, hy1, hx2, hy2 = shot_result["hoop_bbox"]
            hx1,hy1,hx2,hy2 = int(hx1*_shot_sx),int(hy1*_shot_sy),int(hx2*_shot_sx),int(hy2*_shot_sy)
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 128, 255), 1)
        traj_state = shot_result["state"]
        traj_clr   = (0, 255, 100) if traj_state != "WATCHING" else (200, 200, 0)
        cv2.putText(frame, f"TRAJ:{traj_state}",
                    (frame.shape[1] - 160, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, traj_clr, 1)
        # Makes / attempts counter top-right
        makes    = shot_result["makes"]
        attempts = shot_result["attempts"]
        cv2.putText(frame, f"{makes} / {attempts}",
                    (frame.shape[1] - 110, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Handle a confirmed make or miss
    if shot_result is not None and shot_result["attempt"]:
        if shot_result["make"]:
            _score_total      += 1
            _overlay_color     = (0, 255, 0)    # green flash
            _overlay_text      = "Make"
            _fade_counter      = _FADE_FRAMES
            _score_display_cnt = _SCORE_DISPLAY_FRAMES
            print(f"SCORE #{_score_total}  [TRAJ]")
        else:
            _overlay_color = (0, 0, 255)        # red flash
            _overlay_text  = "Miss"
            _fade_counter  = _FADE_FRAMES
            print(f"MISS  (attempts={shot_result['attempts']})")

    # Make / Miss colored screen flash
    if _fade_counter > 0:
        alpha = 0.2 * (_fade_counter / _FADE_FRAMES)
        frame = cv2.addWeighted(frame, 1 - alpha,
                                np.full_like(frame, _overlay_color), alpha, 0)
        # Result text top-right
        (tw, _), _ = cv2.getTextSize(_overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 5)
        cv2.putText(frame, _overlay_text,
                    (frame.shape[1] - tw - 30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, _overlay_color, 5)
        _fade_counter -= 1

    # SCORE! banner (makes only)
    if _score_display_cnt > 0:
        _score_display_cnt -= 1
        ba = min(1.0, _score_display_cnt / 15.0)
        ov = frame.copy()
        cv2.putText(ov, f"SCORE! #{_score_total}",
                    (int(frame.shape[1] * 0.18), int(frame.shape[0] * 0.55)),
                    cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 255, 80), 4)
        cv2.addWeighted(ov, ba, frame, 1 - ba, 0, frame)

    out.write(frame)

cap.release()
out.release()

# Re-encode to H.264 so the file plays in browsers and uploads to social media
subprocess.run(
    ["ffmpeg", "-y", "-i", _TEMP_FILE,
     "-vcodec", "libx264", "-crf", "23",
     "-preset", "fast", "-pix_fmt", "yuv420p",
     "-movflags", "+faststart",
     _OUTPUT_FILE],
    check=True
)
os.remove(_TEMP_FILE)
print(f"Output saved to {_OUTPUT_FILE} (H.264)")