"""HTTP client that publishes detected shots into the FastAPI replay store."""

from __future__ import annotations

import os
import time
from typing import Any

import requests


class ReplayPublisherClient:
    """Publish shot results to the replay API scaffold."""

    def __init__(
        self,
        base_url: str,
        user_id: str,
        device: str,
        fps: int,
        resolution: list[int],
        timeout_s: float = 2.5,
    ):
        self._base_url = base_url.rstrip("/")
        self._user_id = user_id
        self._device = device
        self._fps = fps
        self._resolution = resolution
        self._timeout_s = timeout_s
        self._session_id: str | None = None
        self.last_error: str | None = None

    @property
    def session_id(self) -> str | None:
        return self._session_id

    def ensure_session(self) -> str | None:
        """Ensure backend session exists and return session_id."""
        if self._ensure_session():
            return self._session_id
        return None

    @classmethod
    def from_env(
        cls,
        fps: int,
        resolution: list[int],
    ) -> "ReplayPublisherClient | None":
        """
        Build from env vars.

        Required:
          - PUREARC_API_BASE_URL (example: http://127.0.0.1:8000)

        Optional:
          - PUREARC_API_USER_ID (default: local-user)
          - PUREARC_API_DEVICE (default: backend-runner)
        """
        base_url = os.getenv("PUREARC_API_BASE_URL")
        if not base_url:
            return None

        user_id = os.getenv("PUREARC_API_USER_ID", "local-user")
        device = os.getenv("PUREARC_API_DEVICE", "backend-runner")

        return cls(
            base_url=base_url,
            user_id=user_id,
            device=device,
            fps=fps,
            resolution=resolution,
        )

    def publish_shot(
        self,
        made: bool,
        shot_metrics,
        mistakes,
        dist_result: dict | None = None,
        clip_url: str | None = None,
    ) -> str | None:
        """Publish a shot and return shot_id on success."""
        self.last_error = None

        if not self._ensure_session():
            return None

        payload = {
            "made": made,
            "timestamp_ms": int(time.time() * 1000),
            "metrics": {
                "release_angle": shot_metrics.release_angle,
                "release_height": shot_metrics.release_height,
                "elbow_angle": shot_metrics.elbow_angle,
                "shot_distance_px": shot_metrics.shot_distance_px,
                "arc_height_ratio": shot_metrics.arc_height_ratio,
                "arc_symmetry": shot_metrics.arc_symmetry,
                "knee_elbow_lag": shot_metrics.knee_elbow_lag,
                "shot_tempo": shot_metrics.shot_tempo,
                "torso_drift": shot_metrics.torso_drift,
            },
            "mistakes": [
                {
                    "tag": m.tag,
                    "severity": m.severity.value,
                    "message": m.message,
                    "value": m.value,
                }
                for m in mistakes
            ],
            "context": {
                "distance_bucket": _distance_bucket(dist_result),
            },
            "quality": {
                "frames_used": None,
            },
            "clip_url": clip_url,
        }

        try:
            resp = requests.post(
                f"{self._base_url}/session/{self._session_id}/shots",
                json=payload,
                timeout=self._timeout_s,
            )
            if resp.status_code >= 400:
                self.last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                return None
            data = resp.json()
            return data.get("shot_id")
        except requests.RequestException as exc:
            self.last_error = f"request failed: {exc}"
            return None
        except ValueError:
            self.last_error = "invalid JSON from replay API"
            return None

    def _ensure_session(self) -> bool:
        if self._session_id:
            return True

        body = {
            "user_id": self._user_id,
            "device": self._device,
            "fps": self._fps,
            "resolution": self._resolution,
        }
        try:
            resp = requests.post(
                f"{self._base_url}/session/start",
                json=body,
                timeout=self._timeout_s,
            )
            if resp.status_code >= 400:
                self.last_error = f"session start HTTP {resp.status_code}: {resp.text[:300]}"
                return False
            data = resp.json()
            self._session_id = data.get("session_id")
            if not self._session_id:
                self.last_error = "session start response missing session_id"
                return False
            return True
        except requests.RequestException as exc:
            self.last_error = f"session start failed: {exc}"
            return False
        except ValueError:
            self.last_error = "invalid JSON from session/start"
            return False


def _distance_bucket(dist_result: dict | None) -> str:
    if not dist_result:
        return "unknown"

    dft = dist_result.get("distance_ft")
    if dft is None:
        return "unknown"
    if dft < 8:
        return "close"
    if dft < 16:
        return "mid"
    return "three"
