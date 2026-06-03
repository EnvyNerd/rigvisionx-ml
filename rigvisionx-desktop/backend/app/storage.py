from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import os
import shutil
import json
import sqlite3
import threading
from typing import Optional

from fastapi import UploadFile

UPLOAD_DIR = Path(os.environ.get("RIGVISIONX_UPLOAD_DIR", "data/uploads"))
HISTORY_DB_PATH = Path(os.environ.get("RIGVISIONX_HISTORY_DB", "data/desktop/history.db"))
_HISTORY_LOCK = threading.Lock()


def save_upload(file: UploadFile, *, prefix: str = "dataset") -> Path:
    safe_name = Path(file.filename or f"{prefix}.csv").name.replace(" ", "_")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest = UPLOAD_DIR / f"{timestamp}_{safe_name}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    return dest


def ensure_path(path: Optional[str]) -> Path:
    if not path:
        raise ValueError("Missing data path")
    return Path(path)


def _ensure_history_db() -> None:
    HISTORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(HISTORY_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,
                data_source TEXT,
                metadata TEXT
            )
            """
        )
        conn.commit()


def log_history(action: str, *, data_source: str | None, metadata: dict | None = None) -> None:
    _ensure_history_db()
    record_time = datetime.now(timezone.utc).isoformat()
    payload = json.dumps(metadata or {}, ensure_ascii=True)
    with _HISTORY_LOCK:
        with sqlite3.connect(HISTORY_DB_PATH) as conn:
            conn.execute(
                "INSERT INTO history (timestamp, action, data_source, metadata) VALUES (?, ?, ?, ?)",
                (record_time, action, data_source, payload),
            )
            conn.commit()


def read_history(limit: int = 50) -> list[dict]:
    if limit < 1:
        return []
    if not HISTORY_DB_PATH.exists():
        return []
    _ensure_history_db()
    with _HISTORY_LOCK:
        with sqlite3.connect(HISTORY_DB_PATH) as conn:
            rows = conn.execute(
                "SELECT timestamp, action, data_source, metadata FROM history ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
    rows.reverse()
    records: list[dict] = []
    for timestamp, action, data_source, metadata_json in rows:
        try:
            metadata = json.loads(metadata_json) if metadata_json else {}
        except json.JSONDecodeError:
            metadata = {}
        records.append(
            {
                "timestamp": timestamp,
                "action": action,
                "data_source": data_source,
                "metadata": metadata,
            }
        )
    return records
