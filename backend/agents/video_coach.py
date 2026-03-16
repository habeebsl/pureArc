"""
Video coach — analyzes all shots from a video session and returns 1–5
targeted drill plans.

Uses the AI agent endpoint if configured; otherwise falls back to a
deterministic selection based on the most frequent mistake tags.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any

import requests

from api.models import DrillPlan, ShotDetailResponse


# ── Drill library ─────────────────────────────────────────────────────────────

_DRILL_LIBRARY: dict[str, tuple[int, list[str]]] = {
    "Hand-Off Shooting Drill": (8, [
        "Start near top of key and receive a hand-off.",
        "Take 1–2 rhythm steps into shot.",
        "Repeat for 3 sets of 10 reps.",
    ]),
    "Speed Shooting Drill": (8, [
        "Sprint, stop on balance, and shoot immediately.",
        "Rebound and sprint back each rep.",
        "Complete 3 rounds of 8 makes.",
    ]),
    "Off-the-Dribble Form Shooting": (10, [
        "Use controlled 1–2 footwork off the dribble.",
        "Focus on balanced rise and high finish.",
        "Do 3 sets of 12 pull-up reps.",
    ]),
    "Partner Shooting": (8, [
        "Shooter works at game rhythm while partner rebounds.",
        "Backpedal to spot and catch into shot.",
        "Make 40 total shots.",
    ]),
    "Titan Shooting": (8, [
        "Rotate lines after every rep to add conditioning.",
        "Keep same form under fatigue.",
        "Complete 3 full line cycles.",
    ]),
    "Rainbow Shooting": (8, [
        "Move through multiple spots around the arc.",
        "Emphasize identical release rhythm at each spot.",
        "Complete 2 rainbow cycles.",
    ]),
    "5 Spot Variety Shooting": (10, [
        "Shoot from five spots around the court.",
        "Take different shot types per spot.",
        "Track makes to monitor consistency.",
    ]),
    "31 Shooting Drill": (10, [
        "Alternate inside, mid-range, and 3-point attempts.",
        "Score each make and race to 31 points.",
        "Reset and repeat with the same form cues.",
    ]),
    "One-Hand Form Shooting": (10, [
        "Remove guide hand completely.",
        "Shoot with dominant hand only from 5–7 feet.",
        "Focus on finger-tip release and arc.",
    ]),
    "Balance and Hold Drill": (8, [
        "Hold your release follow-through for 3 seconds.",
        "Check that your hand, elbow, and shoulder are aligned on finish.",
        "Repeat for 3 sets of 10 makes.",
    ]),
}

# Mistake tag → most targeted drill
_TAG_TO_DRILL: dict[str, str] = {
    "flat_arc":             "Off-the-Dribble Form Shooting",
    "low_release":          "One-Hand Form Shooting",
    "arm_shooting":         "One-Hand Form Shooting",
    "elbow_tuck":           "Balance and Hold Drill",
    "rushed_shot":          "Hand-Off Shooting Drill",
    "fading":               "Speed Shooting Drill",
    "torso_drift":          "Speed Shooting Drill",
    "asymmetric_arc":       "Rainbow Shooting",
    "inconsistent_release": "Partner Shooting",
    "flat_trajectory":      "Off-the-Dribble Form Shooting",
}

_DEFAULT_DRILLS = ["5 Spot Variety Shooting", "Partner Shooting", "Balance and Hold Drill"]


# ── Deterministic fallback ────────────────────────────────────────────────────

def build_deterministic_drills(shots: list[ShotDetailResponse]) -> list[DrillPlan]:
    """
    Aggregate mistake tags across all shots and return the top 1–5 most
    targeted drills ordered by frequency of the underlying issue.
    """
    tag_counts: Counter = Counter()
    for shot in shots:
        for mistake in shot.mistakes:
            tag_counts[mistake.tag] += 1

    selected: list[str] = []
    seen: set[str] = set()

    for tag, _ in tag_counts.most_common():
        drill_name = _TAG_TO_DRILL.get(tag)
        if drill_name and drill_name not in seen:
            selected.append(drill_name)
            seen.add(drill_name)
        if len(selected) >= 5:
            break

    # Fill up to at least 1 drill from defaults
    for name in _DEFAULT_DRILLS:
        if len(selected) >= 5:
            break
        if name not in seen:
            selected.append(name)
            seen.add(name)

    return [_build_drill(name) for name in (selected or [_DEFAULT_DRILLS[0]])]


def _build_drill(name: str) -> DrillPlan:
    duration, steps = _DRILL_LIBRARY.get(
        name,
        (8, [
            "Shoot with consistent rhythm and balance.",
            "Track makes and misses each set.",
            "Adjust one cue at a time.",
        ]),
    )
    return DrillPlan(name=name, duration_min=duration, steps=steps, links=[])


# ── AI-enhanced client ────────────────────────────────────────────────────────

class VideoCoachClient:
    """
    Calls the AI coaching agent with all shot data and parses 1–5 DrillPlans.
    Falls back to ``build_deterministic_drills`` if the call fails.
    """

    def __init__(self, base_url: str, api_key: str, timeout_s: float = 30.0):
        self._base_url  = base_url.rstrip("/")
        self._api_key   = api_key
        self._timeout_s = timeout_s
        self.last_error: str | None = None

    @classmethod
    def from_env(cls) -> "VideoCoachClient | None":
        base_url = os.getenv("PUREARC_REPLAY_AGENT_URL")
        api_key  = os.getenv("PUREARC_REPLAY_AGENT_KEY")
        if not base_url or not api_key:
            return None
        timeout_s = float(os.getenv("PUREARC_VIDEO_AGENT_TIMEOUT_SECONDS", "30"))
        return cls(base_url=base_url, api_key=api_key, timeout_s=timeout_s)

    def get_drills(self, shots: list[ShotDetailResponse]) -> list[DrillPlan] | None:
        """
        Ask the AI agent for 1–5 targeted drills based on all shots.
        Returns None on any failure so the caller can use the deterministic
        fallback.
        """
        self.last_error = None

        shot_summaries: list[dict[str, Any]] = []
        for i, shot in enumerate(shots[:20]):
            m = shot.metrics
            shot_summaries.append({
                "shot":             i + 1,
                "made":             shot.made,
                "release_angle":    m.release_angle,
                "elbow_angle":      m.elbow_angle,
                "arc_height_ratio": m.arc_height_ratio,
                "torso_drift":      m.torso_drift,
                "mistakes":         [mk.tag for mk in shot.mistakes],
            })

        payload = {
            "task":         "video_analysis_drills",
            "total_shots":  len(shots),
            "makes":        sum(1 for s in shots if s.made),
            "shots":        shot_summaries,
            "instruction": (
                "Based on the shot data provided, identify the 1 to 5 most impactful "
                "targeted drills for this player. Return a JSON object with a 'drills' "
                "array where each element has: name (string), duration_min (int), "
                "steps (array of strings). Focus on the most repeated mistakes. "
                "Return only valid JSON."
            ),
        }

        body = {
            "model":    "n/a",
            "messages": [{"role": "user", "content": json.dumps(payload)}],
            "stream":   False,
        }
        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        try:
            resp = requests.post(
                f"{self._base_url}/api/v1/chat/completions",
                headers=headers,
                json=body,
                timeout=self._timeout_s,
            )
        except requests.RequestException as exc:
            self.last_error = f"request failed: {exc}"
            return None

        if resp.status_code >= 400:
            self.last_error = f"HTTP {resp.status_code}"
            return None

        try:
            data    = resp.json()
            content = (
                ((data.get("choices") or [{}])[0].get("message") or {})
                .get("content", "")
            )
            if not content:
                self.last_error = "empty agent response"
                return None

            # Extract the first JSON object or array from the text
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if not json_match:
                json_match = re.search(r"\[.*\]", content, re.DOTALL)
            if not json_match:
                self.last_error = "no JSON found in agent response"
                return None

            raw = json.loads(json_match.group())
            drills_raw = raw if isinstance(raw, list) else raw.get("drills") or []
            if not drills_raw:
                return None

            drills: list[DrillPlan] = []
            for d in drills_raw[:5]:
                drills.append(DrillPlan(
                    name         = str(d.get("name", "Form Shooting")),
                    duration_min = int(d.get("duration_min", 8)),
                    steps        = [str(s) for s in (d.get("steps") or [])],
                    links        = [],
                ))
            return drills or None

        except Exception as exc:
            self.last_error = f"parse error: {exc}"
            return None
