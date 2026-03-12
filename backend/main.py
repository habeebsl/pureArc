import cv2
import os
import math
import subprocess
import tempfile
import numpy as np
from detectors import (
    PoseEstimator, POSE_CONNECTIONS,
    ReleaseDetector,
    RimDetector,
    estimate_distance, draw_distance_overlay,
    ShotDetector,
    NetMotionDetector,
    ShotMetricsEngine,
    MistakeEngine,
)
from agents import AsyncLiveCoach, build_live_payload

# RimDetector is kept as fallback when ShotDetector weights are unavailable
_CUSTOM_WEIGHTS = os.path.join(os.path.dirname(__file__), "runs", "rim_detector", "weights", "best.pt")


def draw_landmarks(frame, landmarks, line_color=(0, 255, 0), point_color=(0, 0, 255), thickness=2):
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
                cv2.line(frame, pt1, pt2, line_color, thickness)

    # Draw landmark points
    for lm in landmarks:
        if lm.visibility > 0.5:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, point_color, -1)


# Initialize pose estimator (downloads model on first run)
pose_estimator    = PoseEstimator()
release_detector  = ReleaseDetector()
# RimDetector used only when ShotDetector is unavailable
_rim_detector_fallback = RimDetector(
    custom_model_path=_CUSTOM_WEIGHTS if os.path.exists(_CUSTOM_WEIGHTS) else None
)

# ShotDetector (trajectory-based) — requires trained weights; skipped gracefully if absent
try:
    shot_detector = ShotDetector()
    print("ShotDetector loaded (trajectory scoring active)")
except FileNotFoundError as _e:
    shot_detector = None
    print(f"ShotDetector unavailable: {_e}")

# Net-motion make/miss detector — replaces trajectory-based scoring
net_motion_detector = NetMotionDetector()

# Shot metrics engine — collects per-frame data, computes per-shot metrics
shot_metrics_engine = ShotMetricsEngine()
mistake_engine = MistakeEngine()
live_coach = AsyncLiveCoach.from_env()
_frame_idx = 0

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

# Last known ball position for release detector continuity
_last_ball_xy = None


# ── Ball position smoother ──────────────────────────────────────────────
# Smooths raw YOLO ball detections before feeding them to downstream
# systems (pose selection, release detector).
#
# Problems it solves:
#   • Teleportation — YOLO bbox jumps 300+ px in one frame
#   • Jitter        — bbox center wobbles ±15 px even when ball is still
#   • Staleness     — holding a position forever after ball leaves view
#
# How it works:
#   1. Exponential moving average (alpha = 0.4 → responsive but smooth)
#   2. Outlier rejection: new detection > 120 px from smoothed pos → skip
#   3. Velocity prediction: coast for up to 8 frames after ball is lost
#   4. Staleness: expire after 15 frames of no valid detection
_ball_smooth_xy  = None   # smoothed (x, y) float
_ball_velocity   = (0.0, 0.0)  # (vx, vy) px/frame
_ball_miss_count = 0      # consecutive frames w/o valid YOLO detection
_BALL_SMOOTH_ALPHA   = 0.4   # EMA weight for new detection
_BALL_MAX_JUMP_PX    = 120   # reject detections farther than this from smoothed
_BALL_COAST_FRAMES   = 8     # predict from velocity for this many lost frames
_BALL_EXPIRE_FRAMES  = 15    # kill _last_ball_xy after this many misses


def _update_ball_smooth(raw_xy):
    """
    Call every frame with raw_xy = (x, y) from YOLO or None.
    Updates _ball_smooth_xy, _ball_velocity, _last_ball_xy.
    """
    global _ball_smooth_xy, _ball_velocity, _ball_miss_count, _last_ball_xy

    if raw_xy is not None:
        rx, ry = float(raw_xy[0]), float(raw_xy[1])

        if _ball_smooth_xy is None:
            # First detection — initialise
            _ball_smooth_xy = (rx, ry)
            _ball_velocity  = (0.0, 0.0)
            _ball_miss_count = 0
        else:
            dx = rx - _ball_smooth_xy[0]
            dy = ry - _ball_smooth_xy[1]
            dist = math.hypot(dx, dy)

            if dist <= _BALL_MAX_JUMP_PX:
                # Good detection — blend into EMA
                a = _BALL_SMOOTH_ALPHA
                sx = _ball_smooth_xy[0] * (1 - a) + rx * a
                sy = _ball_smooth_xy[1] * (1 - a) + ry * a
                _ball_velocity  = (sx - _ball_smooth_xy[0], sy - _ball_smooth_xy[1])
                _ball_smooth_xy = (sx, sy)
                _ball_miss_count = 0
            else:
                # Outlier — possibly teleportation.  If we've been missing
                # for several frames, accept it as the ball reappearing.
                if _ball_miss_count >= _BALL_COAST_FRAMES:
                    _ball_smooth_xy = (rx, ry)
                    _ball_velocity  = (0.0, 0.0)
                    _ball_miss_count = 0
                else:
                    # Reject this frame, treat as miss
                    _ball_miss_count += 1
    else:
        _ball_miss_count += 1

    # Coast: predict from velocity while ball is temporarily lost
    if _ball_miss_count > 0 and _ball_smooth_xy is not None:
        if _ball_miss_count <= _BALL_COAST_FRAMES:
            vx, vy = _ball_velocity
            _ball_smooth_xy = (
                _ball_smooth_xy[0] + vx,
                _ball_smooth_xy[1] + vy,
            )

    # Expire
    if _ball_miss_count > _BALL_EXPIRE_FRAMES:
        _ball_smooth_xy = None
        _ball_velocity  = (0.0, 0.0)
        _last_ball_xy   = None
    elif _ball_smooth_xy is not None:
        _last_ball_xy = (int(_ball_smooth_xy[0]), int(_ball_smooth_xy[1]))

# Load video (replace with your file)
cap = cv2.VideoCapture("test_shot8.mp4")

# Set up video writer — write raw mp4v to a temp file, re-encode to H.264 at the end
_OUTPUT_FILE = "output.mp4"
_TEMP_FILE   = "output_tmp.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(_TEMP_FILE, fourcc, 30.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    _frame_idx += 1

    # Keep native frame for shot detector — YOLO handles its own resize internally
    # and distorting portrait videos to 640×480 breaks coordinate geometry.
    _shot_frame = frame
    _shot_sx    = 640 / frame.shape[1]
    _shot_sy    = 480 / frame.shape[0]

    # Resize for performance (optional but recommended)
    frame = cv2.resize(frame, (640, 480))

    # Drain any completed live-coach responses (async, non-blocking)
    if live_coach is not None:
        for msg in live_coach.poll():
            if msg.kind == "coach":
                print("\n--- Live Coach ---")
                print(msg.text)
            else:
                print("\n--- Live Coach Error ---")
                print(f"  {msg.text}")

    # Convert BGR → RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- Run YOLO FIRST so we have fresh ball position for pose selection ---
    shot_result = shot_detector.update(_shot_frame) if shot_detector is not None else None

    # Derive ball position from the shot detector (raw, current-frame).
    _raw_ball_xy = None
    if shot_result is not None and shot_result["ball_bbox"]:
        sx1, sy1, sx2, sy2 = shot_result["ball_bbox"]
        bx = int((sx1 + sx2) / 2 * _shot_sx)
        by = int((sy1 + sy2) / 2 * _shot_sy)
        br = int(max(sx2 - sx1, sy2 - sy1) / 2 * max(_shot_sx, _shot_sy))
        ball_result = (bx, by, br)
        _raw_ball_xy = (bx, by)
    else:
        ball_result = None

    # Update smoother — _last_ball_xy will be the smoothed position
    if ball_result:
        _update_ball_smooth(_raw_ball_xy)
    else:
        _update_ball_smooth(None)

    # For pose selection: prefer raw current-frame ball pos (most accurate
    # for "who is holding the ball"), fall back to smoothed for continuity.
    _pose_ball_xy = _raw_ball_xy if _raw_ball_xy is not None else _last_ball_xy

    # Pass ball position so PoseEstimator picks the ball-handler
    # when multiple people are in frame.
    pose_result = pose_estimator.process_frame(frame_rgb, ball_xy=_pose_ball_xy)
    landmarks = pose_result.primary  # primary player (may be None)

    # When the pose tracker switches to a different person, flush the
    # release detector buffers — stale data from another player's pose
    # causes false positives.
    if pose_estimator.person_switched:
        release_detector.reset()

    if landmarks:
        angles = pose_estimator.get_joint_angles(landmarks)

        print(
            f"Elbow: {angles['elbow_angle']:.2f} | "
            f"Knee: {angles['knee_angle']:.2f}"
        )

        elbow = angles['elbow_angle']

    # Draw all detected skeletons — primary in green, others in gray
    for i, pose_lms in enumerate(pose_result.all_poses):
        if i == pose_result.primary_idx:
            draw_landmarks(frame, pose_lms, line_color=(0, 255, 0), point_color=(0, 0, 255))
        else:
            draw_landmarks(frame, pose_lms, line_color=(130, 130, 130), point_color=(100, 100, 100), thickness=1)

    # Derive hoop position from the shot detector; fall back to RimDetector when unavailable.
    if shot_result is not None and shot_result["hoop_bbox"]:
        hx1, hy1, hx2, hy2 = shot_result["hoop_bbox"]
        # Scale from native resolution to 640×480 and build rim_result dict
        hx1s, hy1s = int(hx1 * _shot_sx), int(hy1 * _shot_sy)
        hx2s, hy2s = int(hx2 * _shot_sx), int(hy2 * _shot_sy)
        rim_result = {
            "center": ((hx1s + hx2s) // 2, (hy1s + hy2s) // 2),
            "bbox":   (hx1s, hy1s, hx2s, hy2s),
            "locked": True,
        }
    else:
        rim_result = _rim_detector_fallback.detect_rim(frame)

    # ── Feed shot metrics engine ─────────────────────────────────────
    _rim_center = rim_result["center"] if rim_result and "center" in rim_result else None
    _angles_for_metrics = angles if landmarks else None
    shot_metrics_engine.feed(
        _frame_idx, landmarks, _last_ball_xy, _rim_center,
        _angles_for_metrics, frame_hw=(480, 640),
    )

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

    if landmarks:
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

        # Ball position: use last known, or wrist coords to zero-out ball signals
        if _last_ball_xy:
            bx, by = _last_ball_xy
        else:
            bx, by = wrist_x, wrist_y
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
            shot_metrics_engine.on_release(_frame_idx)

    # Persist RELEASE text for 30 frames
    if _release_display_cnt > 0:
        _release_display_cnt -= 1
        cv2.putText(frame, "RELEASE", (50, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)

    # ── Net-motion make/miss detection ─────────────────────────────────
    # Feed the net motion detector: it needs the frame, hoop bbox, and
    # whether the ShotDetector is currently ARMED (ball near rim).
    _hoop_bbox_for_net = None
    if rim_result and "bbox" in rim_result:
        _hoop_bbox_for_net = rim_result["bbox"]
    _armed = shot_result is not None and shot_result["state"] == "ARMED"
    net_result = net_motion_detector.update(frame, _hoop_bbox_for_net, _armed)

    # HUD overlays
    if shot_result is not None:
        traj_state = shot_result["state"]
        traj_clr   = (0, 255, 100) if traj_state != "WATCHING" else (200, 200, 0)
        cv2.putText(frame, f"YOLO:{traj_state}",
                    (frame.shape[1] - 160, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, traj_clr, 1)

    # Net motion state + score
    _net_state = net_result["state"]
    _net_score = net_result["net_score"]
    _net_clr = (0, 200, 255) if _net_state == "WATCHING" else (200, 200, 0)
    cv2.putText(frame, f"NET:{_net_state} {_net_score:.1f}",
                (frame.shape[1] - 220, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, _net_clr, 1)

    # Makes / attempts counter top-right (from net motion detector)
    makes    = net_result["makes"]
    attempts = net_result["attempts"]
    cv2.putText(frame, f"{makes} / {attempts}",
                (frame.shape[1] - 110, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Handle a confirmed make or miss from NET MOTION
    if net_result["attempt"]:
        _is_make = net_result["make"]

        # Compute shot metrics for this attempt
        _shot_metrics = shot_metrics_engine.on_result(_frame_idx, made=_is_make)
        if _shot_metrics is not None:
            print(f"\n{'='*50}")
            print(f"SHOT METRICS  ({'MAKE' if _is_make else 'MISS'})")
            print(f"  Release angle:   {_shot_metrics.release_angle}")
            print(f"  Release height:  {_shot_metrics.release_height}")
            print(f"  Elbow angle:     {_shot_metrics.elbow_angle}")
            print(f"  Shot distance:   {_shot_metrics.shot_distance_px}")
            print(f"  Arc peak (px):   {_shot_metrics.arc_peak}")
            print(f"  Arc height ratio:{_shot_metrics.arc_height_ratio}")
            print(f"  Arc symmetry:    {_shot_metrics.arc_symmetry}")
            print(f"  Knee-elbow lag:  {_shot_metrics.knee_elbow_lag} frames")
            print(f"  Shot tempo:      {_shot_metrics.shot_tempo} frames")
            print(f"  Torso drift:     {_shot_metrics.torso_drift} px")
            mistakes = mistake_engine.analyse(_shot_metrics)
            if mistakes:
                print(f"  --- Coaching Cues ---")
                for mk in mistakes:
                    print(f"  [{mk.severity.value:8s}] {mk.tag}: {mk.message}")
            if live_coach is not None:
                payload = build_live_payload(
                    _shot_metrics,
                    mistakes,
                    _is_make,
                    fps=30,
                    dist_result=dist_result,
                )
                if not live_coach.submit(payload):
                    print("  --- Live Coach ---")
                    print("  queue full, skipping this shot's LLM call")
            print(f"{'='*50}\n")

        if _is_make:
            _score_total      += 1
            _overlay_color     = (0, 255, 0)    # green flash
            _overlay_text      = "Make"
            _fade_counter      = _FADE_FRAMES
            _score_display_cnt = _SCORE_DISPLAY_FRAMES
            print(f"SCORE #{_score_total}  [NET]")
        else:
            _overlay_color = (0, 0, 255)        # red flash
            _overlay_text  = "Miss"
            _fade_counter  = _FADE_FRAMES
            print(f"MISS  (attempts={net_result['attempts']})  [NET]")

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
if live_coach is not None:
    live_coach.close()

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