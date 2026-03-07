"""
Distance estimator — player-to-rim floor distance.

Assumptions
-----------
* Side-view or slight-angle camera (not directly behind the shooter).
* The court floor is approximately flat in the image plane.
* Rim height above the floor is exactly 3.05 m (FIBA / NBA standard).
* "Floor level" in the image is approximated by the player's ankle y-pixel.

Two independent pixel-scale methods are computed; if both are available the
result is a weighted average.  Either can be used alone as a fallback.

  Method A — Rim-height scale (preferred)
  ----------------------------------------
  The rim is 3.05 m above the floor.  If the camera is roughly level:
      scale_A  = (ankle_y_px  - rim_y_px) / RIM_HEIGHT_M      [px / m]
      h_dist   = |ankle_x_px  - rim_x_px| / scale_A            [m]

  Method B — Player-height scale (fallback)
  ------------------------------------------
  Uses the pixel distance from the player's nose to their mid-ankle and
  a known / assumed player height.
      scale_B  = (ankle_y_px  - nose_y_px) / player_height_m   [px / m]
      h_dist   = |ankle_x_px  - rim_x_px| / scale_B             [m]

  The floor-level x of the rim is taken directly as rim_x_px — this is
  exact for a pure side-view camera and slightly under-estimates  distance
  for a camera that is elevated and angled down (common in practice).
"""

import math
import numpy as np

# MediaPipe pose landmark indices
_NOSE        = 0
_L_SHOULDER  = 11
_R_SHOULDER  = 12
_L_ANKLE     = 27
_R_ANKLE     = 28
_L_HEEL      = 29
_R_HEEL      = 30

RIM_HEIGHT_M = 3.05   # NBA/FIBA standard


def estimate_distance(
    rim_result:      dict,
    landmarks:       list,
    frame_shape:     tuple,
    player_height_m: float = 1.85,
    weight_a:        float = 0.65,   # weight for rim-height method
    weight_b:        float = 0.35,   # weight for player-height method
) -> dict | None:
    """
    Parameters
    ----------
    rim_result      : dict returned by RimDetector.detect_rim()
    landmarks       : list of NormalizedLandmark from MediaPipe
    frame_shape     : (H, W[, C]) tuple — used to denormalise landmarks
    player_height_m : assumed real-world player height in metres
    weight_a / _b   : blend weights for the two scale methods

    Returns
    -------
    dict with keys:
        "distance_m"   : estimated floor distance in metres (float)
        "foot_px"      : (x, y) pixel position of player's feet
        "rim_px"       : (x, y) pixel position of the rim centre
        "scale_a_ppm"  : px-per-metre from rim-height method (None if unavailable)
        "scale_b_ppm"  : px-per-metre from player-height method (None if unavailable)
        "method"       : "both" | "rim_height" | "player_height"
    or None if insufficient data.
    """
    if rim_result is None or landmarks is None:
        return None

    H, W = frame_shape[:2]
    rim_cx, rim_cy = rim_result["center"]

    # ------------------------------------------------------------------ #
    # Landmark pixel positions
    # ------------------------------------------------------------------ #
    def px(lm):
        return int(lm.x * W), int(lm.y * H)

    nose       = px(landmarks[_NOSE])
    l_ankle    = px(landmarks[_L_ANKLE])
    r_ankle    = px(landmarks[_R_ANKLE])

    # Use the ankle that is most visible (higher confidence)
    l_vis = landmarks[_L_ANKLE].visibility
    r_vis = landmarks[_R_ANKLE].visibility

    if l_vis >= r_vis:
        foot_x, foot_y = l_ankle
    else:
        foot_x, foot_y = r_ankle

    # Average both ankles if both are visible
    if l_vis > 0.5 and r_vis > 0.5:
        foot_x = (l_ankle[0] + r_ankle[0]) // 2
        foot_y = (l_ankle[1] + r_ankle[1]) // 2

    # ------------------------------------------------------------------ #
    # Method A: rim-height scale
    # ------------------------------------------------------------------ #
    scale_a = None
    dist_a  = None
    vertical_px_a = foot_y - rim_cy   # positive when rim is above feet

    if vertical_px_a > 10:            # sanity: rim must be above the feet
        scale_a  = vertical_px_a / RIM_HEIGHT_M          # px per metre
        horiz_px = abs(foot_x - rim_cx)
        dist_a   = horiz_px / scale_a

    # ------------------------------------------------------------------ #
    # Method B: player-height scale
    # ------------------------------------------------------------------ #
    scale_b = None
    dist_b  = None
    vertical_px_b = foot_y - nose[1]  # head (nose) to feet in pixels

    if vertical_px_b > 10:
        scale_b  = vertical_px_b / player_height_m       # px per metre
        horiz_px = abs(foot_x - rim_cx)
        dist_b   = horiz_px / scale_b

    # ------------------------------------------------------------------ #
    # Blend / select
    # ------------------------------------------------------------------ #
    if dist_a is not None and dist_b is not None:
        distance_m = weight_a * dist_a + weight_b * dist_b
        method     = "both"
    elif dist_a is not None:
        distance_m = dist_a
        method     = "rim_height"
    elif dist_b is not None:
        distance_m = dist_b
        method     = "player_height"
    else:
        return None

    return {
        "distance_m":   round(distance_m, 2),
        "distance_ft":  round(distance_m * 3.28084, 1),
        "foot_px":      (foot_x, foot_y),
        "rim_px":       (rim_cx, rim_cy),
        "scale_a_ppm":  round(scale_a, 2) if scale_a else None,
        "scale_b_ppm":  round(scale_b, 2) if scale_b else None,
        "method":       method,
    }


def draw_distance_overlay(frame, dist_result: dict, rim_result: dict) -> None:
    """
    Draws the rim bounding box, a floor-level line to the player's feet,
    and the distance label on *frame* in-place.
    """
    import cv2

    if rim_result is None:
        return

    # Rim bounding box
    x1, y1, x2, y2 = rim_result["bbox"]
    color = (0, 200, 255)   # cyan-orange
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = "RIM" + (" [LOCKED]" if rim_result.get("locked") else "")
    cv2.putText(frame, label, (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    if dist_result is None:
        return

    fx, fy = dist_result["foot_px"]
    rx, ry = dist_result["rim_px"]

    # Dashed floor-level line from feet to rim base
    _draw_dashed_line(frame, (fx, fy), (rx, fy), (255, 200, 0), 1, 10)

    # Vertical line from rim centre down to floor level
    cv2.line(frame, (rx, ry), (rx, fy), color, 1)

    # Distance label
    d_m  = dist_result["distance_m"]
    d_ft = dist_result["distance_ft"]
    cv2.putText(frame,
                f"{d_m:.1f} m  /  {d_ft:.1f} ft",
                ((fx + rx) // 2 - 50, fy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)


def _draw_dashed_line(img, pt1, pt2, color, thickness, dash_len):
    """Draws a dashed line between two points."""
    import cv2

    x1, y1 = pt1
    x2, y2 = pt2
    dist    = math.hypot(x2 - x1, y2 - y1)
    if dist == 0:
        return
    dx = (x2 - x1) / dist
    dy = (y2 - y1) / dist
    i  = 0
    draw = True
    while i < dist:
        x_s = int(x1 + dx * i)
        y_s = int(y1 + dy * i)
        end = min(i + dash_len, dist)
        x_e = int(x1 + dx * end)
        y_e = int(y1 + dy * end)
        if draw:
            cv2.line(img, (x_s, y_s), (x_e, y_e), color, thickness)
        draw = not draw
        i += dash_len
