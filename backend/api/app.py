from __future__ import annotations

import time
from collections import defaultdict

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from .models import (
    FrameIngestResponse,
    ReplayAnalysisRequest,
    ReplayAnalysisResponse,
    SessionStartRequest,
    SessionStartResponse,
    ShotCreateRequest,
    ShotDetailResponse,
    ShotListItem,
    WsEvent,
)
from .replay import build_replay_analysis
from .store import store

app = FastAPI(title="PureArc API", version="0.1.0")

_ws_clients: dict[str, list[WebSocket]] = defaultdict(list)


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "purearc-api"}


@app.post("/session/start", response_model=SessionStartResponse)
def start_session(req: SessionStartRequest) -> SessionStartResponse:
    session = store.create_session(req.user_id, req.device, req.fps, req.resolution)
    return SessionStartResponse(
        session_id=session.session_id,
        ws_url=f"/session/{session.session_id}/events",
    )


@app.post("/session/{session_id}/frame", response_model=FrameIngestResponse)
async def ingest_frame(
    session_id: str,
    frame: UploadFile = File(...),
    timestamp_ms: int | None = None,
) -> FrameIngestResponse:
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")

    _ = await frame.read()
    frame_id = store.ingest_frame(session_id)
    ts = timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)
    return FrameIngestResponse(accepted=True, frame_id=frame_id, timestamp_ms=ts)


@app.post("/session/{session_id}/shots", response_model=ShotDetailResponse)
async def add_shot(session_id: str, req: ShotCreateRequest) -> ShotDetailResponse:
    try:
        shot = store.add_shot(session_id, req)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")

    await _broadcast(
        session_id,
        WsEvent(
            event="shot_result",
            payload={
                "shot_id": shot.shot_id,
                "made": shot.made,
                "timestamp_ms": shot.timestamp_ms,
            },
        ),
    )
    return shot


@app.get("/session/{session_id}/shots", response_model=list[ShotListItem])
def list_shots(session_id: str) -> list[ShotListItem]:
    try:
        return store.list_shots(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")


@app.get("/shots/{shot_id}", response_model=ShotDetailResponse)
def get_shot(shot_id: str) -> ShotDetailResponse:
    shot = store.get_shot(shot_id)
    if shot is None:
        raise HTTPException(status_code=404, detail="shot not found")
    return shot


@app.get("/shots/{shot_id}/clip")
def get_shot_clip(shot_id: str) -> dict:
    shot = store.get_shot(shot_id)
    if shot is None:
        raise HTTPException(status_code=404, detail="shot not found")
    return {
        "shot_id": shot_id,
        "clip_url": shot.clip_url,
        "message": "Clip retrieval scaffolded; wire actual video storage next.",
    }


@app.post("/shots/{shot_id}/replay-analysis", response_model=ReplayAnalysisResponse)
def replay_analysis(shot_id: str, req: ReplayAnalysisRequest) -> ReplayAnalysisResponse:
    shot = store.get_shot(shot_id)
    if shot is None:
        raise HTTPException(status_code=404, detail="shot not found")
    return build_replay_analysis(shot, include_drill=req.include_drill)


@app.websocket("/session/{session_id}/events")
async def session_events(session_id: str, websocket: WebSocket):
    if store.get_session(session_id) is None:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    _ws_clients[session_id].append(websocket)
    try:
        await websocket.send_json({"event": "connected", "payload": {"session_id": session_id}})
        while True:
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _ws_clients[session_id]:
            _ws_clients[session_id].remove(websocket)


async def _broadcast(session_id: str, event: WsEvent):
    clients = list(_ws_clients.get(session_id, []))
    if not clients:
        return

    dead: list[WebSocket] = []
    for ws in clients:
        try:
            await ws.send_json(event.dict())
        except Exception:
            dead.append(ws)

    for ws in dead:
        if ws in _ws_clients[session_id]:
            _ws_clients[session_id].remove(ws)
