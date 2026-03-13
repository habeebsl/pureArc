from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from threading import Lock

from .models import SessionSummary, ShotCreateRequest, ShotDetailResponse, ShotListItem


@dataclass
class SessionState:
    session_id: str
    user_id: str
    device: str
    fps: int
    resolution: list[int]
    created_at_ms: int
    shots: list[str] = field(default_factory=list)
    frame_count: int = 0


class InMemoryStore:
    def __init__(self):
        self._lock = Lock()
        self._sessions: dict[str, SessionState] = {}
        self._shots: dict[str, ShotDetailResponse] = {}

    def create_session(self, user_id: str, device: str, fps: int, resolution: list[int]) -> SessionState:
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        session = SessionState(
            session_id=session_id,
            user_id=user_id,
            device=device,
            fps=fps,
            resolution=resolution,
            created_at_ms=int(time.time() * 1000),
        )
        with self._lock:
            self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> SessionState | None:
        with self._lock:
            return self._sessions.get(session_id)

    def list_sessions(self) -> list[SessionSummary]:
        with self._lock:
            items: list[SessionSummary] = []
            for session in sorted(self._sessions.values(), key=lambda s: s.created_at_ms, reverse=True):
                items.append(
                    SessionSummary(
                        session_id=session.session_id,
                        user_id=session.user_id,
                        device=session.device,
                        fps=session.fps,
                        resolution=session.resolution,
                        created_at_ms=session.created_at_ms,
                        shot_count=len(session.shots),
                    )
                )
            return items

    def latest_session(self) -> SessionSummary | None:
        sessions = self.list_sessions()
        return sessions[0] if sessions else None

    def ingest_frame(self, session_id: str) -> str:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(session_id)
            session.frame_count += 1
            return f"frame_{session.frame_count}"

    def add_shot(self, session_id: str, data: ShotCreateRequest) -> ShotDetailResponse:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(session_id)

            shot_id = f"shot_{uuid.uuid4().hex[:10]}"
            detail = ShotDetailResponse(
                shot_id=shot_id,
                session_id=session_id,
                made=data.made,
                timestamp_ms=data.timestamp_ms,
                metrics=data.metrics,
                mistakes=data.mistakes,
                context=data.context,
                quality=data.quality,
                clip_url=data.clip_url,
            )
            self._shots[shot_id] = detail
            session.shots.append(shot_id)
            return detail

    def list_shots(self, session_id: str) -> list[ShotListItem]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(session_id)

            items: list[ShotListItem] = []
            for shot_id in session.shots:
                shot = self._shots.get(shot_id)
                if shot is None:
                    continue
                items.append(
                    ShotListItem(
                        shot_id=shot.shot_id,
                        made=shot.made,
                        timestamp_ms=shot.timestamp_ms,
                        clip_url=shot.clip_url,
                    )
                )
            return items

    def get_shot(self, shot_id: str) -> ShotDetailResponse | None:
        with self._lock:
            return self._shots.get(shot_id)


store = InMemoryStore()
