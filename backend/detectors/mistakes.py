"""
Mistake Engine — rule-based analysis of ShotMetrics.

Takes a ShotMetrics object and returns a list of Mistake flags describing
what went wrong (or right) with the shot, along with severity and a short
coaching cue.

Usage:
    from detectors.mistakes import MistakeEngine, Mistake

    engine = MistakeEngine()
    mistakes = engine.analyse(shot_metrics)
    for m in mistakes:
        print(f"[{m.severity}] {m.tag}: {m.message}")
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .shot_metrics import ShotMetrics


class Severity(str, Enum):
    INFO    = "info"      # observation, not a problem
    MINOR   = "minor"     # suboptimal but not critical
    MAJOR   = "major"     # likely hurting accuracy
    CRITICAL = "critical" # fundamental mechanic issue


@dataclass
class Mistake:
    tag:      str          # machine-readable label e.g. "flat_arc"
    category: str          # "release" | "arc" | "timing" | "stability"
    severity: Severity
    message:  str          # short coaching cue
    value:    float | None = None  # the metric value that triggered the flag


# ── Thresholds ─────────────────────────────────────────────────────────
# These are based on widely-accepted basketball shooting science
# (e.g. Noah Basketball, Dr. Dish research, and coaching consensus).

# Release angle (degrees) — optimal window shifts with distance
# Close shots (low distance_px): wider acceptable range, lower optimal
# Long shots (high distance_px): tighter window, higher optimal
_RELEASE_ANGLE_LOW      = 35.0    # flat shot territory (absolute floor)
_RELEASE_ANGLE_HIGH     = 65.0    # wasting energy (absolute ceiling)

# Distance-adaptive optimal ranges (pixel distance → angle window)
# These are interpolated: close ≤200px, far ≥400px
_CLOSE_OPTIMAL  = (42.0, 58.0)   # more forgiving for close range
_FAR_OPTIMAL    = (46.0, 54.0)   # tighter for long range
_CLOSE_DIST_PX  = 200.0
_FAR_DIST_PX    = 400.0

# Release height (normalised y, 0=top) — lower is higher on screen
_RELEASE_HEIGHT_LOW     = 0.50    # releasing below mid-frame → low release

# Elbow angle at release (degrees) — should be near full extension
_ELBOW_ANGLE_TUCKED     = 130.0   # arm not extending enough
_ELBOW_ANGLE_OPTIMAL    = (145.0, 175.0)

# Arc peak (pixel y) — lower value = higher on screen = better arc
# Replaced by arc_height_ratio which is distance-normalised.

# Arc height ratio — peak rise / horizontal distance
# Higher = more arc.  Scales with shot length.
_ARC_RATIO_LOW_CLOSE  = 0.25   # close shots don't need as much arc
_ARC_RATIO_LOW_FAR    = 0.35   # long shots need more arc
_ARC_RATIO_HIGH       = 0.90   # excessive arc regardless of distance

# Arc symmetry (0-1, 1 = perfect)
_ARC_SYMMETRY_POOR      = 0.5     # very lopsided trajectory

# Knee-elbow lag (frames, positive = knee first = ideal)
# Negative or zero means arm-only shooting
_KNEE_ELBOW_LAG_MIN     = 1       # knee should extend before elbow

# Shot tempo (frames at 30fps)
_SHOT_TEMPO_FAST        = 6       # ~0.2s — rushed
_SHOT_TEMPO_SLOW        = 30      # ~1.0s — very slow load

# Torso drift (pixels)
_TORSO_DRIFT_HIGH       = 30.0    # significant lean/fade


class MistakeEngine:
    """Analyse a ShotMetrics object and return flagged mistakes."""

    def analyse(self, m: ShotMetrics) -> list[Mistake]:
        """
        Returns a list of Mistake flags for the given shot.
        Empty list = clean shot.
        """
        flags: list[Mistake] = []

        self._check_release_angle(m, flags)
        self._check_release_height(m, flags)
        self._check_elbow_angle(m, flags)
        self._check_arc_height(m, flags)
        self._check_arc_symmetry(m, flags)
        self._check_knee_elbow_timing(m, flags)
        self._check_shot_tempo(m, flags)
        self._check_torso_drift(m, flags)

        return flags

    # ── Individual checks ──────────────────────────────────────────── #

    @staticmethod
    def _optimal_angle_range(dist_px: float | None) -> tuple[float, float]:
        """Interpolate optimal release angle window by shot distance."""
        if dist_px is None:
            return (45.0, 55.0)  # fallback: generic window
        t = max(0.0, min(1.0, (dist_px - _CLOSE_DIST_PX) / (_FAR_DIST_PX - _CLOSE_DIST_PX)))
        lo = _CLOSE_OPTIMAL[0] + t * (_FAR_OPTIMAL[0] - _CLOSE_OPTIMAL[0])
        hi = _CLOSE_OPTIMAL[1] + t * (_FAR_OPTIMAL[1] - _CLOSE_OPTIMAL[1])
        return (lo, hi)

    @staticmethod
    def _check_release_angle(m: ShotMetrics, flags: list[Mistake]):
        if m.release_angle is None:
            return
        a = m.release_angle
        opt = MistakeEngine._optimal_angle_range(m.shot_distance_px)

        if a < _RELEASE_ANGLE_LOW:
            flags.append(Mistake(
                tag="flat_arc",
                category="release",
                severity=Severity.MAJOR,
                message=f"Flat release ({a:.0f}°). Aim for {opt[0]:.0f}-{opt[1]:.0f}° for this distance.",
                value=a,
            ))
        elif a > _RELEASE_ANGLE_HIGH:
            flags.append(Mistake(
                tag="high_arc",
                category="release",
                severity=Severity.MINOR,
                message=f"Release angle too high ({a:.0f}°). Bring it down toward {(opt[0]+opt[1])/2:.0f}°.",
                value=a,
            ))
        elif opt[0] <= a <= opt[1]:
            flags.append(Mistake(
                tag="good_arc",
                category="release",
                severity=Severity.INFO,
                message=f"Good release angle ({a:.0f}°).",
                value=a,
            ))

    @staticmethod
    def _check_release_height(m: ShotMetrics, flags: list[Mistake]):
        if m.release_height is None:
            return
        h = m.release_height

        if h > _RELEASE_HEIGHT_LOW:
            flags.append(Mistake(
                tag="low_release",
                category="release",
                severity=Severity.MAJOR,
                message="Low release point — you're releasing below shoulder height. Extend up through the shot.",
                value=h,
            ))

    @staticmethod
    def _check_elbow_angle(m: ShotMetrics, flags: list[Mistake]):
        if m.elbow_angle is None:
            return
        ea = m.elbow_angle

        if ea < _ELBOW_ANGLE_TUCKED:
            flags.append(Mistake(
                tag="elbow_tucked",
                category="release",
                severity=Severity.MAJOR,
                message=f"Arm not fully extending at release ({ea:.0f}°). Push through and finish high.",
                value=ea,
            ))
        elif _ELBOW_ANGLE_OPTIMAL[0] <= ea <= _ELBOW_ANGLE_OPTIMAL[1]:
            flags.append(Mistake(
                tag="good_extension",
                category="release",
                severity=Severity.INFO,
                message=f"Good arm extension ({ea:.0f}°).",
                value=ea,
            ))

    @staticmethod
    def _check_arc_height(m: ShotMetrics, flags: list[Mistake]):
        if m.arc_height_ratio is None:
            return
        r = m.arc_height_ratio
        # Interpolate the "too low" threshold by distance
        dist = m.shot_distance_px
        if dist is not None:
            t = max(0.0, min(1.0, (dist - _CLOSE_DIST_PX) / (_FAR_DIST_PX - _CLOSE_DIST_PX)))
            low_thresh = _ARC_RATIO_LOW_CLOSE + t * (_ARC_RATIO_LOW_FAR - _ARC_RATIO_LOW_CLOSE)
        else:
            low_thresh = 0.30  # fallback

        if r < low_thresh:
            flags.append(Mistake(
                tag="low_arc",
                category="arc",
                severity=Severity.MAJOR,
                message=f"Arc too flat for this distance (ratio {r:.2f}). Get more height on the ball.",
                value=r,
            ))
        elif r > _ARC_RATIO_HIGH:
            flags.append(Mistake(
                tag="excessive_arc",
                category="arc",
                severity=Severity.MINOR,
                message=f"Arc excessively high (ratio {r:.2f}). You're losing accuracy — flatten it slightly.",
                value=r,
            ))

    @staticmethod
    def _check_arc_symmetry(m: ShotMetrics, flags: list[Mistake]):
        if m.arc_symmetry is None:
            return
        s = m.arc_symmetry

        if s < _ARC_SYMMETRY_POOR:
            flags.append(Mistake(
                tag="asymmetric_arc",
                category="arc",
                severity=Severity.MINOR,
                message=f"Lopsided trajectory (symmetry {s:.2f}). Could indicate a push shot or line drive.",
                value=s,
            ))

    @staticmethod
    def _check_knee_elbow_timing(m: ShotMetrics, flags: list[Mistake]):
        if m.knee_elbow_lag is None:
            return
        lag = m.knee_elbow_lag

        if lag < _KNEE_ELBOW_LAG_MIN:
            flags.append(Mistake(
                tag="arm_shooting",
                category="timing",
                severity=Severity.MAJOR,
                message="Arm-only shot — legs aren't driving the ball. Extend legs before the arm.",
                value=lag,
            ))
        else:
            flags.append(Mistake(
                tag="good_sequencing",
                category="timing",
                severity=Severity.INFO,
                message=f"Good leg-to-arm sequencing ({lag} frame lead).",
                value=lag,
            ))

    @staticmethod
    def _check_shot_tempo(m: ShotMetrics, flags: list[Mistake]):
        if m.shot_tempo is None:
            return
        t = m.shot_tempo

        if t < _SHOT_TEMPO_FAST:
            flags.append(Mistake(
                tag="rushed_shot",
                category="timing",
                severity=Severity.MINOR,
                message=f"Rushed release ({t} frames / ~{t/30:.1f}s). Slow down and set your feet.",
                value=t,
            ))
        elif t > _SHOT_TEMPO_SLOW:
            flags.append(Mistake(
                tag="slow_load",
                category="timing",
                severity=Severity.MINOR,
                message=f"Slow shot load ({t} frames / ~{t/30:.1f}s). Quicker release = harder to contest.",
                value=t,
            ))

    @staticmethod
    def _check_torso_drift(m: ShotMetrics, flags: list[Mistake]):
        if m.torso_drift is None:
            return
        d = m.torso_drift

        if d > _TORSO_DRIFT_HIGH:
            flags.append(Mistake(
                tag="fading",
                category="stability",
                severity=Severity.MAJOR,
                message=f"Fading/leaning during shot ({d:.0f}px drift). Stay balanced through the release.",
                value=d,
            ))
