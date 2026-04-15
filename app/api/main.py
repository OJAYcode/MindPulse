from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.db.database import initialize_database
from app.utils.config import get_settings


def create_app() -> FastAPI:
    initialize_database()
    settings = get_settings()
    app = FastAPI(title="Stress Detection API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.frontend_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
