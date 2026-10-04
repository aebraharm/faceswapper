"""FastAPI entry point. Run with: uvicorn app.main:app --host 0.0.0.0 --port 8000"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.api.routes import router as api_router
from app.camera.websocket import router as websocket_router
from app.config import Settings
from app.services import create_services


def create_app(settings: Settings | None = None) -> FastAPI:
    application = FastAPI(
        title="Real-time Face AI",
        description="Local-first, opt-in face analysis and model-pluggable transformation API.",
        version="0.1.0",
    )
    application.state.services = create_services(settings)
    application.include_router(api_router)
    application.include_router(websocket_router)

    @application.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    return application


app = create_app()
