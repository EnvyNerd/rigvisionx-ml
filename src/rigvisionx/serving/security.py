from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from typing import Any

import bcrypt
import jwt


JWT_SECRET = os.environ.get("RIGVISIONX_JWT_SECRET", "rigvisionx-dev-secret")
JWT_ALGORITHM = os.environ.get("RIGVISIONX_JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = int(os.environ.get("RIGVISIONX_JWT_EXPIRE_MINUTES", "480"))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(*, subject: str, role: str) -> str:
    expires = _utcnow() + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {
        "sub": subject,
        "role": role,
        "exp": expires,
        "iat": _utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])

