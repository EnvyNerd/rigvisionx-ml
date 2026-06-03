from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src" / "rigvisionx").exists():
            return parent
    return here.parents[3]


def _ensure_repo_src_on_path() -> None:
    repo_root = _find_repo_root()
    src_path = repo_root / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_ensure_repo_src_on_path()


def _load_desktop_env() -> None:
    desktop_root = Path(__file__).resolve().parents[2]
    env_path = desktop_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


_load_desktop_env()

from .routes import router


def create_app() -> FastAPI:
    app = FastAPI(title="TerraEnergy AI Desktop Backend")
    app.include_router(router)

    allowed_origins = os.environ.get("RIGVISIONX_ALLOWED_ORIGINS")
    if allowed_origins:
        origins = [item.strip() for item in allowed_origins.split(",") if item.strip()]
        if origins:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

    if os.environ.get("RIGVISIONX_API_KEY=rigvisionx-dev-key") != "0":
        try:
            from rigvisionx.serving.api import app as core_app
        except Exception:
            core_app = None
        if core_app is not None:
            app.mount("/core", core_app)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("RIGVISIONX_HOST", "127.0.0.1")
    port = int(os.environ.get("RIGVISIONX_PORT", "8081"))
    uvicorn.run(app, host=host, port=port)

