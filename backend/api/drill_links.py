from __future__ import annotations

import os
import urllib.parse
from typing import Any

import requests


DEFAULT_TOPIC = "basketball shooting drills"


def fetch_drill_links(drill_titles: list[str], max_results_per_drill: int = 2) -> tuple[dict[str, list[str]], str, list[str]]:
    """
    Fetch YouTube links from deployed serverless function.

    Returns:
      - mapping drill_title -> [url, ...]
      - provider name
      - errors
    """
    clean_titles = [t.strip() for t in drill_titles if t and t.strip()]
    if not clean_titles:
        return {}, "none", ["no drill titles provided"]

    fn_url = os.getenv("PUREARC_YT_FUNCTION_URL")
    if not fn_url:
        return _fallback(clean_titles), "fallback_query_links", ["PUREARC_YT_FUNCTION_URL not configured"]

    body = {
        "drill_titles": clean_titles,
        "topic": os.getenv("PUREARC_YT_TOPIC", DEFAULT_TOPIC),
        "max_results_per_drill": max_results_per_drill,
    }

    timeout_s = float(os.getenv("PUREARC_YT_TIMEOUT_SECONDS", "8.0"))
    try:
        resp = requests.post(fn_url, json=body, timeout=timeout_s)
    except requests.RequestException as exc:
        return _fallback(clean_titles), "fallback_query_links", [f"function request failed: {exc}"]

    if resp.status_code >= 400:
        return _fallback(clean_titles), "fallback_query_links", [f"function http {resp.status_code}"]

    try:
        payload = resp.json()
    except ValueError:
        return _fallback(clean_titles), "fallback_query_links", ["function invalid json"]

    results = payload.get("results") or []
    out: dict[str, list[str]] = {}
    for item in results:
        drill = str(item.get("drill_title") or "").strip()
        if not drill:
            continue
        links = []
        for v in item.get("videos") or []:
            url = str(v.get("url") or "").strip()
            if url:
                links.append(url)
        out[drill] = links

    if not out:
        return _fallback(clean_titles), "fallback_query_links", ["function returned no links"]

    provider = str(payload.get("provider") or "youtube_function")
    errors = [str(e) for e in (payload.get("errors") or [])]
    return out, provider, errors


def _fallback(drill_titles: list[str]) -> dict[str, list[str]]:
    topic = os.getenv("PUREARC_YT_TOPIC", DEFAULT_TOPIC)
    out: dict[str, list[str]] = {}
    for drill in drill_titles:
        q = urllib.parse.quote_plus(f"{drill} {topic}")
        out[drill] = [f"https://www.youtube.com/results?search_query={q}"]
    return out
