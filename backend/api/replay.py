from __future__ import annotations

from .drill_links import fetch_drill_links
from .models import DrillPlan, ReplayAnalysisResponse, ShotDetailResponse


def build_replay_analysis(shot: ShotDetailResponse, include_drill: bool) -> ReplayAnalysisResponse:
    went_well: list[str] = []
    to_fix: list[str] = []

    if shot.made:
        went_well.append("You made the shot — keep reinforcing this form.")

    m = shot.metrics

    if m.release_angle is not None and 45 <= m.release_angle <= 55:
        went_well.append(f"Release angle is strong at {m.release_angle:.1f}°.")
    elif m.release_angle is not None:
        to_fix.append(f"Release angle ({m.release_angle:.1f}°) is outside the ideal window for consistency.")

    if m.elbow_angle is not None and 145 <= m.elbow_angle <= 175:
        went_well.append(f"Arm extension looks good at release ({m.elbow_angle:.1f}°).")
    elif m.elbow_angle is not None:
        to_fix.append(f"Arm extension at release ({m.elbow_angle:.1f}°) can be improved.")

    if m.arc_height_ratio is not None and m.arc_height_ratio >= 0.35:
        went_well.append(f"Arc shape is healthy (ratio {m.arc_height_ratio:.2f}).")
    elif m.arc_height_ratio is not None:
        to_fix.append(f"Arc is a bit flat (ratio {m.arc_height_ratio:.2f}); add more lift.")

    if m.torso_drift is not None and m.torso_drift > 30:
        to_fix.append(f"Torso drift is high ({m.torso_drift:.1f}px); focus on balance.")

    if shot.mistakes:
        for mk in shot.mistakes[:3]:
            to_fix.append(mk.message)

    if not went_well:
        went_well.append("Good rep captured — use replay to reinforce rhythm and finish.")

    if not to_fix:
        to_fix.append("No major issues flagged on this shot.")

    drill = None
    backup_drill = None
    links_provider = None
    links_errors: list[str] = []
    if include_drill:
        primary_name, backup_name = _pick_drill_names(shot)
        title_order = [primary_name] + ([backup_name] if backup_name else [])
        links_map, links_provider, links_errors = fetch_drill_links(title_order, max_results_per_drill=2)

        drill = _build_drill_plan(primary_name, links_map.get(primary_name, []))
        if backup_name:
            backup_drill = _build_drill_plan(backup_name, links_map.get(backup_name, []))

    recommendations = [
        "Track 10-shot trends instead of one-shot outcomes.",
        "Prioritize repeatable release timing and balanced landings.",
        "Re-check mechanics at game speed after each drill block.",
    ]

    return ReplayAnalysisResponse(
        shot_id=shot.shot_id,
        what_went_well=went_well[:4],
        what_to_fix=to_fix[:5],
        drill=drill,
        backup_drill=backup_drill,
        links_provider=links_provider,
        links_errors=links_errors,
        general_recommendations=recommendations,
    )


def _pick_drill_names(shot: ShotDetailResponse) -> tuple[str, str | None]:
    # First priority: map explicit mistake tags
    tags = [m.tag for m in shot.mistakes]

    if "flat_arc" in tags:
        return "Off-the-Dribble Form Shooting", "Partner Shooting"
    if "rushed_shot" in tags:
        return "Hand-Off Shooting Drill", "Speed Shooting Drill"
    if "fading" in tags:
        return "Speed Shooting Drill", "Off-the-Dribble Form Shooting"
    if "arm_shooting" in tags:
        return "Partner Shooting", "Titan Shooting"
    if "asymmetric_arc" in tags:
        return "Rainbow Shooting", "Partner Shooting"

    # Fallback from metric thresholds
    m = shot.metrics
    if m.arc_height_ratio is not None and m.arc_height_ratio < 0.35:
        return "Off-the-Dribble Form Shooting", "Partner Shooting"
    if m.torso_drift is not None and m.torso_drift > 30:
        return "Speed Shooting Drill", "Partner Shooting"

    return "5 Spot Variety Shooting", "31 Shooting Drill"


def _build_drill_plan(name: str, links: list[str]) -> DrillPlan:
    library: dict[str, tuple[int, list[str]]] = {
        "Hand-Off Shooting Drill": (
            8,
            [
                "Start near top of key and receive a hand-off.",
                "Take 1-2 rhythm steps into shot.",
                "Repeat for 3 sets of 10 reps.",
            ],
        ),
        "Speed Shooting Drill": (
            8,
            [
                "Sprint, stop on balance, and shoot immediately.",
                "Rebound and sprint back each rep.",
                "Complete 3 rounds of 8 makes.",
            ],
        ),
        "Off-the-Dribble Form Shooting": (
            10,
            [
                "Use controlled 1-2 footwork off dribble.",
                "Focus on balanced rise and high finish.",
                "Do 3 sets of 12 pull-up reps.",
            ],
        ),
        "Partner Shooting": (
            8,
            [
                "Shooter works at game rhythm while partner rebounds.",
                "Backpedal to spot and catch into shot.",
                "Make 40 total shots.",
            ],
        ),
        "Titan Shooting": (
            8,
            [
                "Rotate lines after every rep to add conditioning.",
                "Keep same form under fatigue.",
                "Complete 3 full line cycles.",
            ],
        ),
        "Rainbow Shooting": (
            8,
            [
                "Move through multiple spots around the arc.",
                "Emphasize identical release rhythm at each spot.",
                "Complete 2 rainbow cycles.",
            ],
        ),
        "5 Spot Variety Shooting": (
            10,
            [
                "Shoot from five spots around court.",
                "Take different shot types per spot.",
                "Track makes to monitor consistency.",
            ],
        ),
        "31 Shooting Drill": (
            10,
            [
                "Alternate inside, mid-range, and 3-point attempts.",
                "Score each make and race to 31 points.",
                "Reset and repeat with same form cues.",
            ],
        ),
    }

    duration, steps = library.get(
        name,
        (
            8,
            [
                "Shoot with consistent rhythm and balance.",
                "Track makes and misses each set.",
                "Adjust one cue at a time.",
            ],
        ),
    )

    return DrillPlan(name=name, duration_min=duration, steps=steps, links=links)
