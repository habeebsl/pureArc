"""Client and payload helpers for the live coaching agent."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import requests


_DEFAULT_TIMEOUT_S = 20


class LiveCoachClient:
    """Simple HTTP client for the DigitalOcean agent endpoint."""

    def __init__(self, base_url: str, api_key: str, timeout_s: int = _DEFAULT_TIMEOUT_S):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_s = timeout_s
        self.last_error: str | None = None

    @classmethod
    def from_env(cls) -> "LiveCoachClient | None":
        """Build a client from env vars or return None if missing."""
        base_url = os.getenv("PUREARC_LIVE_AGENT_URL")
        api_key = os.getenv("PUREARC_LIVE_AGENT_KEY")
        if not base_url or not api_key:
            return None
        return cls(base_url, api_key)

    def get_coaching(self, payload: dict[str, Any]) -> str | None:
        """Send a payload and return the assistant response text, if any."""
        self.last_error = None
        url = f"{self._base_url}/api/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        body = {
            "messages": [
                {"role": "user", "content": json.dumps(payload)}
            ],
        }

        try:
            resp = requests.post(url, headers=headers, json=body, timeout=self._timeout_s)
            if resp.status_code >= 400:
                self.last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                return None
            data = resp.json()
        except requests.RequestException as exc:
            self.last_error = f"Request failed: {exc}"
            return None
        except ValueError:
            self.last_error = "Invalid JSON response from live agent"
            return None

        choices = data.get("choices") or []
        if not choices:
            self.last_error = "No choices in live agent response"
            return None
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if not content:
            self.last_error = "Empty message content from live agent"
            return None
        return _sanitize_content(content)


def build_live_payload(
    shot_metrics,
    mistakes,
    made: bool,
    fps: int,
    dist_result: dict | None = None,
) -> dict[str, Any]:
    """Build a compact payload for the live coach agent."""
    payload: dict[str, Any] = {
        "made": made,
        "fps": fps,
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
    }

    if dist_result and "distance_ft" in dist_result:
        payload["context"] = {
            "distance_bucket": _distance_bucket(dist_result["distance_ft"])
        }

    return payload


def _distance_bucket(distance_ft: float) -> str:
    """Map a distance in feet to a bucket label."""
    if distance_ft < 8:
        return "close"
    if distance_ft < 16:
        return "mid"
    return "three"


def _sanitize_content(text: str) -> str:
    """Strip reasoning tags and return clean coach-facing text."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()
