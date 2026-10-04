"""FastAPI entry point. For local development use 127.0.0.1; Electron uses launcher.py."""
from __future__ import annotations

import hmac
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

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
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["null", "file://", "http://127.0.0.1:5173", "http://localhost:5173"],
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-Frame-Session", "X-Frame-Shutdown-Token"],
        allow_credentials=False,
    )

    @application.middleware("http")
    async def enforce_desktop_session(request: Request, call_next):
        expected = os.getenv("FRAME_DESKTOP_SESSION_TOKEN", "")
        if expected and request.method != "OPTIONS":
            supplied = request.headers.get("x-frame-session", "")
            if not hmac.compare_digest(expected, supplied):
                return JSONResponse({"detail": "Not found."}, status_code=404)
        return await call_next(request)

    @application.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    return application


app = create_app()
