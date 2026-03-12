"""Asynchronous wrapper for live coach agent calls."""

from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import Any

from .live_coach import LiveCoachClient


@dataclass
class LiveCoachMessage:
    """A completed async coaching result."""
    kind: str  # "coach" | "error"
    text: str


class AsyncLiveCoach:
    """Background queue worker for non-blocking live-coach requests."""

    def __init__(self, client: LiveCoachClient, max_queue: int = 4):
        self._client = client
        self._jobs: Queue[dict[str, Any] | None] = Queue(maxsize=max_queue)
        self._results: Queue[LiveCoachMessage] = Queue()
        self._stop = Event()
        self._thread = Thread(target=self._run, name="live-coach-worker", daemon=True)
        self._thread.start()

    @classmethod
    def from_env(cls, max_queue: int = 4) -> "AsyncLiveCoach | None":
        client = LiveCoachClient.from_env()
        if client is None:
            return None
        return cls(client, max_queue=max_queue)

    def submit(self, payload: dict[str, Any]) -> bool:
        """Queue a coaching job; returns False if queue is full."""
        try:
            self._jobs.put_nowait(payload)
            return True
        except Full:
            return False

    def poll(self) -> list[LiveCoachMessage]:
        """Drain all currently available results."""
        out: list[LiveCoachMessage] = []
        while True:
            try:
                out.append(self._results.get_nowait())
            except Empty:
                break
        return out

    def close(self, timeout_s: float = 1.5):
        """Stop background worker and wait briefly for shutdown."""
        self._stop.set()
        try:
            self._jobs.put_nowait(None)
        except Full:
            pass
        self._thread.join(timeout=timeout_s)

    def _run(self):
        while not self._stop.is_set():
            try:
                job = self._jobs.get(timeout=0.2)
            except Empty:
                continue

            if job is None:
                break

            text = self._client.get_coaching(job)
            if text:
                self._results.put(LiveCoachMessage(kind="coach", text=text))
            elif self._client.last_error:
                self._results.put(LiveCoachMessage(kind="error", text=self._client.last_error))
