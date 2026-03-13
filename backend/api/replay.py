from __future__ import annotations

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
    if include_drill:
        drill = _pick_drill(shot)

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
        general_recommendations=recommendations,
    )


def _pick_drill(shot: ShotDetailResponse) -> DrillPlan:
    metrics = shot.metrics

    if metrics.arc_height_ratio is not None and metrics.arc_height_ratio < 0.35:
        return DrillPlan(
            name="One-Hand Arc Control",
            duration_min=8,
            steps=[
                "Stand 6-8 ft from rim and shoot one-handed with guide hand off ball.",
                "Focus on lifting elbow and finishing high with wrist snap.",
                "Complete 3 sets of 12 reps, logging arc consistency.",
            ],
        )

    if metrics.torso_drift is not None and metrics.torso_drift > 30:
        return DrillPlan(
            name="Balance Stick Landing",
            duration_min=8,
            steps=[
                "Shoot from mid-range and hold landing for 2 seconds each rep.",
                "Keep nose, chest, and hips stacked toward target.",
                "Complete 3 sets of 10 reps; restart rep if landing drifts.",
            ],
        )

    return DrillPlan(
        name="Form-to-Game-Speed Ladder",
        duration_min=10,
        steps=[
            "5 close form shots, 5 mid-range rhythm shots, 5 game-speed shots.",
            "Keep same release cues across all phases.",
            "Repeat ladder twice and compare make rate + consistency.",
        ],
    )
