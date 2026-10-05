from __future__ import annotations

import re
import threading
import time
from copy import deepcopy
from typing import Any


_POST_ID_RE = re.compile(r"threads\.(?:com|net)/@[^/?#]+/post/([^/?#]+)", re.IGNORECASE)
_LOCK = threading.RLock()
_PROGRESS: dict[str, dict[str, Any]] = {}


def extract_post_id(url: str) -> str | None:
    match = _POST_ID_RE.search(url)
    return match.group(1) if match else None


def set_analysis_progress(
    post_id: str,
    *,
    state: str,
    message: str,
    current: int | None = None,
    total: int | None = None,
) -> dict[str, Any]:
    now = time.time()
    with _LOCK:
        previous = _PROGRESS.get(post_id, {})
        started_at = float(previous.get("started_at") or now)
        payload = {
            "post_id": post_id,
            "state": state,
            "message": message,
            "current": current if current is not None else previous.get("current", 0),
            "total": total if total is not None else previous.get("total", 0),
            "started_at": started_at,
            "updated_at": now,
        }
        _PROGRESS[post_id] = payload
        return deepcopy(payload)


def start_analysis_progress(post_id: str, total: int) -> dict[str, Any]:
    with _LOCK:
        _PROGRESS.pop(post_id, None)
    return set_analysis_progress(
        post_id,
        state="starting",
        message="正在啟動 Threads Collector……",
        current=0,
        total=total,
    )


def get_analysis_progress(post_id: str) -> dict[str, Any] | None:
    now = time.time()
    with _LOCK:
        payload = deepcopy(_PROGRESS.get(post_id))
    if payload is None:
        return None
    payload["elapsed_seconds"] = max(0, int(now - float(payload["started_at"])))
    payload["idle_seconds"] = max(0, int(now - float(payload["updated_at"])))
    return payload
