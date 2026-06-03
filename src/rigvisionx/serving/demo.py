from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _bool_env(name: str, default: str = "false") -> bool:
    raw = os.environ.get(name, default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _resolve_path(raw_path: str | None, default_rel: str) -> Path:
    if raw_path:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        return candidate
    return REPO_ROOT / default_rel


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _load_license_file(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    if not text:
        return False
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return True
    if isinstance(payload, dict):
        if payload.get("licensed") is True:
            return True
        if str(payload.get("license_key", "")).strip():
            return True
    return False


def _is_licensed() -> bool:
    if _bool_env("RIGVISIONX_LICENSED", "false"):
        return True
    if os.environ.get("RIGVISIONX_LICENSE_KEY", "").strip():
        return True
    license_file_raw = os.environ.get("RIGVISIONX_LICENSE_FILE", "").strip()
    if license_file_raw:
        license_path = _resolve_path(license_file_raw, "data/license.json")
        return _load_license_file(license_path)
    return False


def _get_demo_start(state_path: Path, now: datetime) -> datetime:
    payload = _load_json(state_path)
    start_value = payload.get("start_utc") if isinstance(payload, dict) else None
    start_dt = _parse_datetime(start_value)
    if start_dt is None:
        start_dt = now
        _write_json(
            state_path,
            {
                "start_utc": start_dt.isoformat(),
                "created_at_utc": now.isoformat(),
            },
        )
    return start_dt


def get_demo_status(now: datetime | None = None) -> dict[str, Any]:
    demo_mode = _bool_env("RIGVISIONX_DEMO_MODE", "false")
    demo_days = max(1, _int_env("RIGVISIONX_DEMO_DAYS", 7))
    state_path = _resolve_path(
        os.environ.get("RIGVISIONX_DEMO_STATE_PATH"), "data/demo_state.json"
    )

    status: dict[str, Any] = {
        "demo_mode": demo_mode,
        "demo_days": demo_days,
        "licensed": _is_licensed(),
    }

    if not demo_mode:
        return status

    now = now or _utcnow()
    start_dt = _get_demo_start(state_path, now)
    expires_dt = start_dt + timedelta(days=demo_days)
    remaining_seconds = (expires_dt - now).total_seconds()
    days_remaining = int((remaining_seconds + 86399) // 86400) if remaining_seconds > 0 else 0
    expired = now >= expires_dt

    status.update(
        {
            "start_utc": start_dt.isoformat(),
            "expires_utc": expires_dt.isoformat(),
            "expired": expired,
            "days_remaining": days_remaining,
            "state_path": str(state_path),
        }
    )

    return status
