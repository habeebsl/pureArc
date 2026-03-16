from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from agents.video_coach import VideoCoachClient, build_deterministic_drills
from agents.replay_coach import ReplayCoachClient
from video_processor import VideoProcessor, ShotEvent

logger = logging.getLogger("purearc.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

from .models import (
    DrillPlan,
    MistakePayload,
    ReplayAnalysisRequest,
    ReplayAnalysisResponse,
    ShotContextPayload,
    ShotCreateRequest,
    ShotDetailResponse,
    ShotMetricsPayload,
    VideoAnalysisResponse,
)
from .replay import build_replay_analysis
from .store import store

app = FastAPI(title="PureArc API", version="0.2.0")
_BACKEND_ROOT = Path(__file__).resolve().parent.parent

_video_coach  = VideoCoachClient.from_env()
_replay_coach = ReplayCoachClient.from_env()

# One worker — video processing is CPU-bound
_executor = ThreadPoolExecutor(max_workers=1)


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "purearc-api"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _event_to_create_request(event: ShotEvent, ts: int) -> ShotCreateRequest:
    m = event.metrics
    return ShotCreateRequest(
        made=event.made,
        timestamp_ms=ts,
        metrics=ShotMetricsPayload(
            release_angle=m.release_angle,
            release_height=m.release_height,
            elbow_angle=m.elbow_angle,
            shot_distance_px=m.shot_distance_px,
            arc_height_ratio=m.arc_height_ratio,
            arc_symmetry=m.arc_symmetry,
            knee_elbow_lag=m.knee_elbow_lag,
            shot_tempo=m.shot_tempo,
            torso_drift=m.torso_drift,
        ),
        mistakes=[
            MistakePayload(
                tag=mk.tag,
                severity=mk.severity.value,
                message=mk.message,
                value=mk.value,
            )
            for mk in event.mistakes
        ],
    )


def _run_processor(video_path: str) -> list[ShotEvent]:
    """Blocking video processing — called inside thread-pool executor."""
    proc = VideoProcessor()
    return proc.process_video(video_path)


# ── Video analysis ────────────────────────────────────────────────────────────

@app.post("/analyze-video", response_model=VideoAnalysisResponse)
async def analyze_video(video: UploadFile = File(...)) -> VideoAnalysisResponse:
    """
    Upload a video file; detect every shot attempt; return metrics + 1-5 drills.
    Processing runs at ~15 fps equivalent. A 30-second clip takes 1-3 min on CPU.
    """
    raw    = await video.read()
    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        loop   = asyncio.get_event_loop()
        events = await loop.run_in_executor(_executor, _run_processor, tmp_path)
    except Exception:
        logger.exception("Video processing failed")
        raise HTTPException(status_code=500, detail="Video processing failed")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    session    = store.create_session(user_id="upload", device="video", fps=30, resolution=[640, 480])
    session_id = session.session_id
    ts_base    = int(time.time() * 1000)

    shots: list[ShotDetailResponse] = []
    for i, event in enumerate(events):
        req  = _event_to_create_request(event, ts_base + i * 1000)
        shot = store.add_shot(session_id, req)
        shots.append(shot)

    makes = sum(1 for s in shots if s.made)
    total = len(shots)

    drills: list[DrillPlan] | None = None
    if _video_coach is not None:
        drills = _video_coach.get_drills(shots)
        if not drills:
            logger.warning("VideoCoach failed: %s", _video_coach.last_error)

    if not drills:
        drills = build_deterministic_drills(shots)

    if total == 0:
        summary = (
            "No shot attempts were detected. Make sure the hoop and shooter "
            "are both visible and try a shorter clip."
        )
    else:
        pct     = round(makes / total * 100)
        summary = (
            f"{makes} of {total} shot{'s' if total != 1 else ''} made "
            f"({pct}% FG) detected in this clip."
        )

    logger.info("analyze_video: session=%s shots=%d makes=%d drills=%d",
                session_id, total, makes, len(drills))

    return VideoAnalysisResponse(
        session_id=session_id,
        total_shots=total,
        makes=makes,
        shots=shots,
        drills=drills,
        summary=summary,
    )


# ── Shot retrieval ────────────────────────────────────────────────────────────

@app.get("/shots/{shot_id}", response_model=ShotDetailResponse)
def get_shot(shot_id: str) -> ShotDetailResponse:
    shot = store.get_shot(shot_id)
    if shot is None:
        raise HTTPException(status_code=404, detail="shot not found")
    return shot


@app.get("/shots/{shot_id}/clip")
def get_shot_clip(shot_id: str):
    shot = store.get_shot(shot_id)
    if shot is None:
        raise HTTPException(status_code=404, detail="shot not found")
    if not shot.clip_url:
        raise HTTPException(status_code=404, detail="clip not available for this shot")

    clip_path = Path(shot.clip_url)
    if not clip_path.is_absolute():
        clip_path = (_BACKEND_ROOT / clip_path).resolve()

    if not clip_path.exists() or not clip_path.is_file():
        raise HTTPException(status_code=404, detail="clip file not found")

    return FileResponse(str(clip_path), media_type="video/mp4", filename=clip_path.name)


@app.post("/shots/{shot_id}/replay-analysis", response_model=ReplayAnalysisResponse)
def replay_analysis(shot_id: str, req: ReplayAnalysisRequest) -> ReplayAnalysisResponse:
    shot = store.get_shot(shot_id)
    if shot is None:
        raise HTTPException(status_code=404, detail="shot not found")

    base = build_replay_analysis(shot, include_drill=req.include_drill)

    if _replay_coach is not None:
        enhanced = _replay_coach.enhance(shot=shot, deterministic=base, detail_level=req.detail_level)
        if enhanced is not None:
            return enhanced

    return base
