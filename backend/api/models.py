from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DistanceBucket(str, Enum):
    close = "close"
    mid = "mid"
    three = "three"
    unknown = "unknown"


class SessionStartRequest(BaseModel):
    user_id: str
    device: str = "unknown"
    fps: int = 30
    resolution: list[int] = Field(default_factory=lambda: [1280, 720])


class SessionStartResponse(BaseModel):
    session_id: str
    ws_url: str


class SessionSummary(BaseModel):
    session_id: str
    user_id: str
    device: str
    fps: int
    resolution: list[int]
    created_at_ms: int
    shot_count: int


class SessionLatestResponse(BaseModel):
    session: SessionSummary | None = None


class FrameIngestResponse(BaseModel):
    accepted: bool
    frame_id: str
    timestamp_ms: int


class MistakePayload(BaseModel):
    tag: str
    severity: str
    message: str
    value: float | None = None


class ShotMetricsPayload(BaseModel):
    release_angle: float | None = None
    release_height: float | None = None
    elbow_angle: float | None = None
    shot_distance_px: float | None = None
    arc_height_ratio: float | None = None
    arc_symmetry: float | None = None
    knee_elbow_lag: float | None = None
    shot_tempo: int | None = None
    torso_drift: float | None = None


class ShotContextPayload(BaseModel):
    distance_bucket: DistanceBucket = DistanceBucket.unknown
    shot_type: str | None = None
    dominant_hand: str | None = None
    confidence: float | None = None


class ShotQualityPayload(BaseModel):
    ball_track_frames: int | None = None
    pose_visibility_pct: float | None = None
    frames_used: int | None = None


class ShotCreateRequest(BaseModel):
    made: bool
    timestamp_ms: int
    metrics: ShotMetricsPayload
    mistakes: list[MistakePayload] = Field(default_factory=list)
    context: ShotContextPayload = Field(default_factory=ShotContextPayload)
    quality: ShotQualityPayload = Field(default_factory=ShotQualityPayload)
    clip_url: str | None = None


class ShotListItem(BaseModel):
    shot_id: str
    made: bool
    timestamp_ms: int
    clip_url: str | None = None


class ShotDetailResponse(BaseModel):
    shot_id: str
    session_id: str
    made: bool
    timestamp_ms: int
    metrics: ShotMetricsPayload
    mistakes: list[MistakePayload]
    context: ShotContextPayload
    quality: ShotQualityPayload
    clip_url: str | None = None


class ReplayAnalysisRequest(BaseModel):
    include_drill: bool = True
    detail_level: str = "high"


class DrillPlan(BaseModel):
    name: str
    duration_min: int
    steps: list[str]


class ReplayAnalysisResponse(BaseModel):
    shot_id: str
    what_went_well: list[str]
    what_to_fix: list[str]
    drill: DrillPlan | None = None
    general_recommendations: list[str]


class WsEvent(BaseModel):
    event: str
    payload: dict[str, Any]
