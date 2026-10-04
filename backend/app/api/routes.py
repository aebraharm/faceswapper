"""Validated HTTP routes for local camera, source, model and performance state."""
from __future__ import annotations

import hmac
import os

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from app.api.schemas import CameraSettingsRequest, CameraStartRequest, SourceFaceSelectionRequest, TargetSelectionRequest, TransformerLoadRequest
from app.detection.image_validation import ImageValidationError, decode_source_image
from app.output.video import output_capabilities
from app.transformation.base import ModelNotConfiguredError

router = APIRouter()


def services(request: Request):
    return request.app.state.services


@router.get("/health")
def health(request: Request) -> dict[str, object]:
    state = services(request)
    return {
        "status": "ok",
        "service": "realtime-face-transform",
        "landmarks_available": state.analyzer.landmarks_available,
        "transformer_loaded": state.transformer.loaded,
        "privacy": "source images and identity features are held in memory only",
    }


@router.get("/camera/devices")
def camera_devices() -> dict[str, object]:
    # Browsers intentionally own camera permission and device enumeration.
    return {
        "devices": [],
        "source": "browser",
        "requires_browser_permission": True,
        "message": "Grant camera permission in this page, then choose a device from the browser-provided list.",
    }


@router.post("/camera/start")
def start_camera(body: CameraStartRequest, request: Request) -> dict[str, object]:
    state = services(request)
    state.camera.start(body.device_id)
    return {"ok": True, **state.camera.status()}


@router.post("/camera/stop")
def stop_camera(request: Request) -> dict[str, object]:
    state = services(request)
    state.camera.stop()
    return {"ok": True, **state.camera.status()}


@router.get("/camera/status")
def camera_status(request: Request) -> dict[str, object]:
    return services(request).camera.status()


@router.post("/camera/target")
def select_target(body: TargetSelectionRequest, request: Request) -> dict[str, object]:
    state = services(request)
    if body.face_index is not None and body.face_index >= state.camera.status()["face_count"]:
        raise HTTPException(status_code=422, detail="That target face is no longer visible. Choose a detected face.")
    try:
        state.camera.select_target(body.face_index)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"ok": True, **state.camera.status()}


@router.post("/camera/settings")
def camera_settings(body: CameraSettingsRequest, request: Request) -> dict[str, object]:
    state = services(request)
    if body.transform_enabled:
        model_ready = state.transformer.loaded and state.source_faces.status()["model_ready"]
        if not model_ready:
            raise HTTPException(
                status_code=409,
                detail="Load a compatible face model and select a source face before enabling transformation.",
            )
    try:
        updated = state.camera.configure(
            transform_enabled=body.transform_enabled,
            intensity=body.intensity,
            processing_resolution=body.processing_resolution,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"ok": True, **updated}


@router.post("/source-face/upload")
async def upload_source_face(request: Request) -> dict[str, object]:
    state = services(request)
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    upload_limit_mb = state.settings.max_upload_bytes / (1024 * 1024)
    too_large_message = f"Image is too large. Maximum upload size is {upload_limit_mb:g} MB."
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > state.settings.max_upload_bytes:
                raise HTTPException(status_code=413, detail=too_large_message)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header.") from None
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > state.settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail=too_large_message)
        chunks.append(chunk)
    content = b"".join(chunks)
    try:
        decoded = decode_source_image(
            content,
            content_type,
            max_bytes=state.settings.max_upload_bytes,
            min_dimension=state.settings.min_image_dimension,
            max_dimension=state.settings.max_image_dimension,
            max_pixels=state.settings.max_image_pixels,
        )
    except ImageValidationError as exc:
        message = str(exc)
        code = 413 if "upload is too large" in message.lower() else 415 if ("unsupported image type" in message.lower() or "format" in message.lower()) else 422
        raise HTTPException(status_code=code, detail=message) from exc
    try:
        source_status = state.source_faces.upload(decoded.rgb)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not align/prepare the selected source face: {exc}") from exc
    return {**source_status, "format": decoded.format}


@router.get("/source-face/status")
def source_face_status(request: Request) -> dict[str, object]:
    return services(request).source_faces.status()


@router.post("/source-face/select")
def select_source_face(body: SourceFaceSelectionRequest, request: Request) -> dict[str, object]:
    try:
        return services(request).source_faces.select_face(body.face_index)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not prepare the selected source face: {exc}") from exc


@router.delete("/source-face")
def delete_source_face(request: Request) -> dict[str, object]:
    state = services(request)
    state.source_faces.clear()
    state.camera.configure(transform_enabled=False)
    return {"ok": True, **state.source_faces.status()}


@router.post("/transformer/load")
def load_transformer(body: TransformerLoadRequest, request: Request) -> dict[str, object]:
    state = services(request)
    try:
        result = state.transformer.load(body.provider or state.settings.default_provider)
        state.source_faces.prepare_for_loaded_model()
        return result
    except ModelNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not load the configured model bundle: {exc}") from exc


@router.post("/transformer/unload")
def unload_transformer(request: Request) -> dict[str, object]:
    state = services(request)
    state.camera.configure(transform_enabled=False)
    state.transformer.unload()
    state.source_faces.clear_model_representation()
    return state.transformer.status()


@router.get("/transformer/status")
def transformer_status(request: Request) -> dict[str, object]:
    return services(request).transformer.status()


@router.get("/performance")
def performance(request: Request) -> dict[str, object]:
    state = services(request)
    return {
        **state.camera.status(),
        "model": state.transformer.status(),
        "landmarks_available": state.analyzer.landmarks_available,
        "output": output_capabilities(),
    }


@router.post("/internal/shutdown", include_in_schema=False)
def request_desktop_shutdown(request: Request, background_tasks: BackgroundTasks) -> dict[str, bool]:
    """Authenticated sidecar shutdown hook used only by the Electron parent process."""
    expected = os.getenv("FRAME_DESKTOP_SHUTDOWN_TOKEN", "")
    supplied = request.headers.get("x-frame-shutdown-token", "")
    if not expected or not hmac.compare_digest(expected, supplied):
        # Do not disclose the endpoint to unrelated local browser pages.
        raise HTTPException(status_code=404, detail="Not found.")
    shutdown = getattr(request.app.state, "request_desktop_shutdown", None)
    if shutdown is None:
        raise HTTPException(status_code=503, detail="Desktop shutdown hook is unavailable.")
    background_tasks.add_task(shutdown)
    return {"ok": True}
