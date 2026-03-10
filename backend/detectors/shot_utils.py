"""
Shot detection utility functions, ported from Avi Shah's basketball tracker.
All functions operate on the shared position-list format:
  ball_pos  — list of ((cx, cy), frame_count, w, h, conf)
  hoop_pos  — list of ((cx, cy), frame_count, w, h, conf)
"""

import math
import numpy as np


def score(ball_pos, hoop_pos):
    """
    Return True if the ball's trajectory (linear interpolation) crosses the
    rim's top plane within the rim's horizontal bounds.
    Operates on the last crossing from above → below the rim height.
    """
    rim_height = hoop_pos[-1][0][1] - 0.5 * hoop_pos[-1][3]
    x, y = [], []

    for i in reversed(range(len(ball_pos))):
        if ball_pos[i][0][1] < rim_height:
            x.append(ball_pos[i][0][0])
            y.append(ball_pos[i][0][1])
            if i + 1 < len(ball_pos):
                x.append(ball_pos[i + 1][0][0])
                y.append(ball_pos[i + 1][0][1])
            break

    if len(x) > 1:
        m, b = np.polyfit(x, y, 1)
        predicted_x = (rim_height - b) / m
        rim_x1 = hoop_pos[-1][0][0] - 0.45 * hoop_pos[-1][2]
        rim_x2 = hoop_pos[-1][0][0] + 0.45 * hoop_pos[-1][2]
        rebound = 15
        if rim_x1 - rebound < predicted_x < rim_x2 + rebound:
            return True

    # Fallback: only one point found above the rim — if it's horizontally
    # within the rim bounds the ball almost certainly went through
    if len(x) == 1:
        rim_x1 = hoop_pos[-1][0][0] - 0.45 * hoop_pos[-1][2]
        rim_x2 = hoop_pos[-1][0][0] + 0.45 * hoop_pos[-1][2]
        if rim_x1 - 15 < x[0] < rim_x2 + 15:
            return True

    return False


def detect_down(ball_pos, hoop_pos):
    """True when the ball is below the net (hoop bottom edge)."""
    y = hoop_pos[-1][0][1] + 0.5 * hoop_pos[-1][3]
    return ball_pos[-1][0][1] > y


def detect_up(ball_pos, hoop_pos):
    """True when the ball is in the upper backboard region — a shot is in flight."""
    hx, hy = hoop_pos[-1][0]
    hw, hh = hoop_pos[-1][2], hoop_pos[-1][3]
    x1 = hx - 4 * hw
    x2 = hx + 4 * hw
    y1 = hy - 2 * hh
    y2 = hy - 0.5 * hh
    bx, by = ball_pos[-1][0]
    return x1 < bx < x2 and y1 < by < y2


def in_hoop_region(center, hoop_pos):
    """True if center is within 1× the hoop bbox — used to lower ball conf threshold."""
    if not hoop_pos:
        return False
    hx, hy = hoop_pos[-1][0]
    hw, hh = hoop_pos[-1][2], hoop_pos[-1][3]
    x1, x2 = hx - hw, hx + hw
    y1, y2 = hy - hh, hy + 0.5 * hh
    cx, cy = center
    return x1 < cx < x2 and y1 < cy < y2


def clean_ball_pos(ball_pos, frame_count, hoop_pos=None):
    """
    Remove erratic or stale ball positions:
    - Jump > 4× diameter within 5 frames → spurious detection
      (exempted when the new detection is near the hoop — the ball
       legitimately jumps from shooter to rim area during a shot arc)
    - Non-square bbox (aspect ratio > 1.4) → probably not the ball
    - Positions older than 30 frames
    """
    if len(ball_pos) > 1:
        (x1, y1), f1, w1, h1, _ = ball_pos[-2]
        (x2, y2), f2, w2, h2, _ = ball_pos[-1]
        f_dif = f2 - f1
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        max_dist = 4 * math.sqrt(w1 ** 2 + h1 ** 2)

        # Don't reject near-hoop detections — during a shot the ball
        # jumps from the shooter area to the rim area in a few frames.
        near_hoop = hoop_pos and in_hoop_region((x2, y2), hoop_pos)

        if dist > max_dist and f_dif < 5 and not near_hoop:
            ball_pos.pop()
        elif (w2 * 1.4 < h2) or (h2 * 1.4 < w2):
            ball_pos.pop()

    if ball_pos and frame_count - ball_pos[0][1] > 30:
        ball_pos.pop(0)

    return ball_pos


def clean_hoop_pos(hoop_pos):
    """
    Stabilise hoop position:
    - Jump > 0.5× diameter within 5 frames → bad detection
    - Non-square bbox (aspect ratio > 1.3) → reject
    - Keep only last 25 positions
    """
    if len(hoop_pos) > 1:
        (x1, y1), f1, w1, h1, _ = hoop_pos[-2]
        (x2, y2), f2, w2, h2, _ = hoop_pos[-1]
        f_dif = f2 - f1
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        max_dist = 0.5 * math.sqrt(w1 ** 2 + h1 ** 2)

        if dist > max_dist and f_dif < 5:
            hoop_pos.pop()
        elif (w2 * 1.3 < h2) or (h2 * 1.3 < w2):
            hoop_pos.pop()

    if len(hoop_pos) > 25:
        hoop_pos.pop(0)

    return hoop_pos
