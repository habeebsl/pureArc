from __future__ import annotations

import os
import urllib.parse
from typing import Any

import requests


YOUTUBE_SEARCH_API = "https://www.googleapis.com/youtube/v3/search"
DEFAULT_TOPIC = "basketball shooting drills"
DEFAULT_RESULTS_PER_DRILL = 2
MAX_RESULTS_PER_DRILL = 3


def _is_web_invocation(args: dict[str, Any]) -> bool:
    return "__ow_method" in args or "__ow_path" in args or "__ow_headers" in args


def _to_http(payload: dict[str, Any], status_code: int = 200) -> dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": payload,
    }


def _to_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    return [text] if text else []


def _query_link(query: str) -> str:
    return f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(query)}"


def _fallback_links(drill_titles: list[str], topic: str, results_per_drill: int) -> dict[str, Any]:
    results = []
    for drill in drill_titles:
        query = f"{drill} {topic}".strip()
        videos = [
            {
                "title": f"Search YouTube: {query}",
                "url": _query_link(query),
                "channel": None,
                "published_at": None,
                "thumbnail": None,
                "source": "fallback_query_link",
            }
            for _ in range(results_per_drill)
        ]
        results.append({"drill_title": drill, "videos": videos})

    return {
        "ok": True,
        "provider": "fallback_query_links",
        "results": results,
        "errors": [],
        "note": "YOUTUBE_API_KEY not configured; returning deterministic YouTube search links.",
    }


def _search_youtube(api_key: str, drill: str, topic: str, results_per_drill: int, region_code: str | None) -> tuple[list[dict[str, Any]], str | None]:
    query = f"{drill} {topic}".strip()

    params = {
        "part": "snippet",
        "type": "video",
        "maxResults": results_per_drill,
        "q": query,
        "key": api_key,
        "safeSearch": "strict",
        "videoEmbeddable": "true",
    }
    if region_code:
        params["regionCode"] = region_code

    try:
        response = requests.get(YOUTUBE_SEARCH_API, params=params, timeout=6)
    except requests.RequestException as exc:
        return [], f"request failed for '{drill}': {exc}"

    if response.status_code >= 400:
        return [], f"youtube api error for '{drill}': HTTP {response.status_code}"

    try:
        payload = response.json()
    except ValueError:
        return [], f"invalid json from youtube api for '{drill}'"

    items = payload.get("items") or []
    videos: list[dict[str, Any]] = []
    for item in items:
        vid = ((item.get("id") or {}).get("videoId"))
        snippet = item.get("snippet") or {}
        if not vid:
            continue
        videos.append(
            {
                "title": snippet.get("title"),
                "url": f"https://www.youtube.com/watch?v={vid}",
                "channel": snippet.get("channelTitle"),
                "published_at": snippet.get("publishedAt"),
                "thumbnail": ((snippet.get("thumbnails") or {}).get("high") or {}).get("url"),
                "source": "youtube_data_api",
            }
        )

    if not videos:
        return [], f"no videos found for '{drill}'"

    return videos, None


def main(args: dict[str, Any]) -> dict[str, Any]:
    """
    DigitalOcean Function entrypoint.

    Input args:
      - drill_titles: list[str] (required)
      - topic: str (optional, default "basketball shooting drills")
      - max_results_per_drill: int (optional, default 2, max 3)
      - region_code: str (optional, e.g. "US")

    Output:
      {
        "ok": bool,
        "provider": "youtube_data_api" | "fallback_query_links",
        "results": [
          {"drill_title": "...", "videos": [{"title":..., "url":...}, ...]}
        ],
        "errors": [...]
      }
    """
    drill_titles = _to_list(args.get("drill_titles"))
    topic = str(args.get("topic") or DEFAULT_TOPIC).strip()

    max_results_raw = args.get("max_results_per_drill", DEFAULT_RESULTS_PER_DRILL)
    try:
        max_results = int(max_results_raw)
    except (TypeError, ValueError):
        max_results = DEFAULT_RESULTS_PER_DRILL
    max_results = max(1, min(MAX_RESULTS_PER_DRILL, max_results))

    region_code = args.get("region_code")
    if region_code is not None:
        region_code = str(region_code).strip() or None

    if not drill_titles:
        result = {
            "ok": False,
            "provider": None,
            "results": [],
            "errors": ["drill_titles is required and must be a non-empty list"],
        }
        return _to_http(result, status_code=400) if _is_web_invocation(args) else result

    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        result = _fallback_links(drill_titles, topic, max_results)
        return _to_http(result) if _is_web_invocation(args) else result

    results = []
    errors = []

    for drill in drill_titles:
        videos, err = _search_youtube(api_key, drill, topic, max_results, region_code)
        if err:
            errors.append(err)
            query = f"{drill} {topic}".strip()
            videos = [
                {
                    "title": f"Search YouTube: {query}",
                    "url": _query_link(query),
                    "channel": None,
                    "published_at": None,
                    "thumbnail": None,
                    "source": "fallback_query_link",
                }
            ]
        results.append({"drill_title": drill, "videos": videos})

    result = {
        "ok": True,
        "provider": "youtube_data_api",
        "results": results,
        "errors": errors,
    }
    return _to_http(result) if _is_web_invocation(args) else result
