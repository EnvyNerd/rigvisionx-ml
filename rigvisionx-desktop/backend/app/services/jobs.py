from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock, Thread
from typing import Any, Optional
from uuid import uuid4

from .train import run_training

_JOBS: dict[str, dict[str, Any]] = {}
_LOCK = Lock()


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_job(payload: dict[str, Any]) -> dict[str, Any]:
    job_id = uuid4().hex
    record = {
        "job_id": job_id,
        "status": "queued",
        "created_at": _utcnow(),
        "updated_at": _utcnow(),
        "payload": payload,
        "result": None,
        "error": None,
    }
    with _LOCK:
        _JOBS[job_id] = record
    return record


def update_job(job_id: str, *, status: str, result: Optional[dict] = None, error: Optional[str] = None) -> None:
    with _LOCK:
        record = _JOBS.get(job_id)
        if record is None:
            return
        record["status"] = status
        record["updated_at"] = _utcnow()
        if result is not None:
            record["result"] = result
        if error is not None:
            record["error"] = error


def get_job(job_id: str) -> Optional[dict[str, Any]]:
    with _LOCK:
        record = _JOBS.get(job_id)
        return dict(record) if record else None


def list_jobs(limit: int = 50) -> list[dict[str, Any]]:
    with _LOCK:
        records = list(_JOBS.values())
    records.sort(key=lambda item: item.get("created_at") or "")
    return records[-limit:]


def start_training_job(payload: dict[str, Any]) -> dict[str, Any]:
    record = create_job(payload)
    job_id = record["job_id"]

    def _runner() -> None:
        update_job(job_id, status="running")
        try:
            result = run_training(**payload)
            update_job(job_id, status="completed", result=result)
        except Exception as exc:
            update_job(job_id, status="failed", error=str(exc))

    Thread(target=_runner, daemon=True).start()
    return record
