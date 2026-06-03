from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text, create_engine, func, select
from sqlalchemy.orm import Session, declarative_base, sessionmaker


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_db_path() -> Path:
    raw_path = os.environ.get("RIGVISIONX_DB_PATH")
    if raw_path:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
    else:
        candidate = REPO_ROOT / "data" / "rigvisionx.db"

    try:
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate
    except PermissionError:
        fallback = Path.home() / ".rigvisionx" / "rigvisionx.db"
        fallback.parent.mkdir(parents=True, exist_ok=True)
        return fallback


DB_PATH = _resolve_db_path()

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(120), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(32), default="viewer", nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    last_login_at = Column(DateTime(timezone=True), nullable=True)


class TrainJobRecord(Base):
    __tablename__ = "train_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(64), unique=True, index=True, nullable=False)
    status = Column(String(32), nullable=False)
    training_type = Column(String(32), nullable=False)
    data_source = Column(Text, nullable=False)
    save_models = Column(Boolean, default=True, nullable=False)
    output_dir = Column(Text, nullable=False)
    window_size = Column(Integer, default=24, nullable=False)
    created_by = Column(String(120), nullable=True)
    progress_json = Column(Text, default="[]", nullable=False)
    result_json = Column(Text, nullable=True)
    error_text = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)


class DatasetRecord(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(String(64), unique=True, index=True, nullable=False)
    filename = Column(String(255), nullable=False)
    path = Column(Text, nullable=False)
    source = Column(String(32), default="upload", nullable=False)
    content_type = Column(String(120), nullable=True)
    size_bytes = Column(Integer, nullable=True)
    uploaded_by = Column(String(120), nullable=True)
    uploaded_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_session() -> Session:
    return SessionLocal()


def _encode_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True)


def _decode_json(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def count_users(db: Session) -> int:
    stmt = select(func.count()).select_from(User)
    value = db.scalar(stmt)
    return int(value or 0)


def get_user_by_username(db: Session, username: str) -> User | None:
    stmt = select(User).where(User.username == username)
    return db.scalar(stmt)


def create_user(db: Session, *, username: str, password_hash: str, role: str) -> User:
    user = User(username=username, password_hash=password_hash, role=role)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def update_user_login(db: Session, user: User) -> None:
    user.last_login_at = _utcnow()
    db.add(user)
    db.commit()


def create_train_job(
    db: Session,
    *,
    job_id: str,
    status: str,
    training_type: str,
    data_source: str,
    save_models: bool,
    output_dir: str,
    window_size: int,
    created_by: str | None,
) -> TrainJobRecord:
    record = TrainJobRecord(
        job_id=job_id,
        status=status,
        training_type=training_type,
        data_source=data_source,
        save_models=save_models,
        output_dir=output_dir,
        window_size=window_size,
        created_by=created_by,
        progress_json="[]",
        updated_at=_utcnow(),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_train_job(db: Session, job_id: str) -> TrainJobRecord | None:
    stmt = select(TrainJobRecord).where(TrainJobRecord.job_id == job_id)
    return db.scalar(stmt)


def update_train_job_progress(
    db: Session, job: TrainJobRecord, progress_items: Iterable[dict[str, Any]]
) -> None:
    job.progress_json = _encode_json(list(progress_items))
    job.updated_at = _utcnow()
    db.add(job)
    db.commit()


def finalize_train_job(
    db: Session,
    job: TrainJobRecord,
    *,
    status: str,
    result: dict[str, Any] | None,
    error_text: str | None,
    progress_items: Iterable[dict[str, Any]],
) -> None:
    job.status = status
    job.result_json = _encode_json(result) if result is not None else None
    job.error_text = error_text
    job.progress_json = _encode_json(list(progress_items))
    job.updated_at = _utcnow()
    db.add(job)
    db.commit()


def decode_progress(job: TrainJobRecord) -> list[dict[str, Any]]:
    return _decode_json(job.progress_json, [])


def decode_result(job: TrainJobRecord) -> dict[str, Any] | None:
    return _decode_json(job.result_json, None)


def create_dataset_record(
    db: Session,
    *,
    dataset_id: str,
    filename: str,
    path: str,
    source: str,
    content_type: str | None,
    size_bytes: int | None,
    uploaded_by: str | None,
) -> DatasetRecord:
    record = DatasetRecord(
        dataset_id=dataset_id,
        filename=filename,
        path=path,
        source=source,
        content_type=content_type,
        size_bytes=size_bytes,
        uploaded_by=uploaded_by,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def list_datasets(db: Session) -> list[DatasetRecord]:
    stmt = select(DatasetRecord).order_by(DatasetRecord.uploaded_at.desc())
    return list(db.scalars(stmt))
