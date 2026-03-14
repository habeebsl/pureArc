from __future__ import annotations

import json
import os
import re
from typing import Any

import requests

from api.models import DrillPlan, MomentAnnotation, ReplayAnalysisResponse, ShotDetailResponse


class ReplayCoachClient:
    """Optional client for a deployed replay analysis agent endpoint."""

    def __init__(self, base_url: str, api_key: str, timeout_s: float = 12.0):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_s = timeout_s
        self.last_error: str | None = None

    @classmethod
    def from_env(cls) -> "ReplayCoachClient | None":
        base_url = os.getenv("PUREARC_REPLAY_AGENT_URL")
        api_key = os.getenv("PUREARC_REPLAY_AGENT_KEY")
        if not base_url or not api_key:
            return None
        timeout_s = float(os.getenv("PUREARC_REPLAY_AGENT_TIMEOUT_SECONDS", "12"))
        return cls(base_url=base_url, api_key=api_key, timeout_s=timeout_s)

    def enhance(
        self,
        shot: ShotDetailResponse,
        deterministic: ReplayAnalysisResponse,
        detail_level: str = "high",
    ) -> ReplayAnalysisResponse | None:
        """Call replay agent and map its JSON output back into ReplayAnalysisResponse."""
        self.last_error = None

        body = {
            "model": "n/a",
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "shot": shot.dict(),
                            "deterministic_analysis": deterministic.dict(),
                            "detail_level": detail_level,
                        }
                    ),
                }
            ],
            "stream": False,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        try:
            resp = requests.post(
                f"{self._base_url}/api/v1/chat/completions",
                headers=headers,
                json=body,
                timeout=self._timeout_s,
            )
        except requests.RequestException as exc:
            self.last_error = f"request failed: {exc}"
            return None

        if resp.status_code >= 400:
            self.last_error = f"HTTP {resp.status_code}: {resp.text[:240]}"
            return None

        try:
            data = resp.json()
            content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
            if not content:
                self.last_error = "empty content from replay agent"
                return None
            parsed = _parse_json_content(content)
            if parsed is None:
                self.last_error = "replay agent returned non-JSON content"
                return None
            return _merge_response(deterministic, parsed)
        except ValueError:
            self.last_error = "invalid json from replay agent"
            return None


def _parse_json_content(content: str) -> dict[str, Any] | None:
    text = content.strip()

    # Try raw JSON first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except ValueError:
        pass

    # Try fenced JSON
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except ValueError:
            return None

    return None


def _to_drill(obj: dict[str, Any] | None, fallback: DrillPlan | None) -> DrillPlan | None:
    if not obj:
        return fallback

    name = str(obj.get("name") or (fallback.name if fallback else "")).strip()
    if not name:
        return fallback

    duration = obj.get("duration_min", fallback.duration_min if fallback else 8)
    try:
        duration = int(duration)
    except (TypeError, ValueError):
        duration = fallback.duration_min if fallback else 8

    steps_raw = obj.get("steps") or (fallback.steps if fallback else [])
    steps = [str(s).strip() for s in steps_raw if str(s).strip()]
    if not steps and fallback:
        steps = fallback.steps

    links_raw = obj.get("links") or (fallback.links if fallback else [])
    links = [str(u).strip() for u in links_raw if str(u).strip()]

    return DrillPlan(name=name, duration_min=duration, steps=steps, links=links)


def _to_moment_annotations(raw: Any, fallback: list[MomentAnnotation]) -> list[MomentAnnotation]:
    if not isinstance(raw, list):
        return fallback

    out: list[MomentAnnotation] = []
    for item in raw:
        if not isinstance(item, dict):
            continue

        try:
            t_sec = float(item.get("t_sec"))
            frame_idx = int(item.get("frame_idx"))
        except (TypeError, ValueError):
            continue

        tag = str(item.get("tag") or "").strip()
        observation = str(item.get("observation") or "").strip()
        correction = str(item.get("correction") or "").strip()
        if not tag or not observation or not correction:
            continue

        out.append(
            MomentAnnotation(
                t_sec=t_sec,
                frame_idx=frame_idx,
                tag=tag,
                observation=observation,
                correction=correction,
            )
        )

    if not out:
        return fallback
    return out[:6]


def _merge_response(base: ReplayAnalysisResponse, llm_obj: dict[str, Any]) -> ReplayAnalysisResponse:
    went_well = llm_obj.get("what_went_well") or base.what_went_well
    to_fix = llm_obj.get("what_to_fix_first") or llm_obj.get("what_to_fix") or base.what_to_fix
    recs = llm_obj.get("general_recommendations") or base.general_recommendations
    moment_annotations = _to_moment_annotations(llm_obj.get("moment_annotations"), base.moment_annotations)

    next_shot_focus = llm_obj.get("next_shot_focus")
    if isinstance(next_shot_focus, str):
        next_shot_focus = next_shot_focus.strip() or None
    else:
        next_shot_focus = None

    if next_shot_focus is None:
        next_shot_focus = base.next_shot_focus

    if "summary" in llm_obj and isinstance(llm_obj["summary"], str) and llm_obj["summary"].strip():
        # Keep UI concise; include summary as first recommendation line.
        recs = [llm_obj["summary"].strip()] + [r for r in recs if r]

    merged = ReplayAnalysisResponse(
        shot_id=base.shot_id,
        moment_annotations=moment_annotations,
        what_went_well=[str(x) for x in went_well][:4],
        what_to_fix=[str(x) for x in to_fix][:5],
        drill=_to_drill(llm_obj.get("primary_drill"), base.drill),
        backup_drill=_to_drill(llm_obj.get("backup_drill"), base.backup_drill),
        links_provider=base.links_provider,
        links_errors=base.links_errors,
        general_recommendations=[str(x) for x in recs][:5],
        next_shot_focus=next_shot_focus,
    )

    return merged
