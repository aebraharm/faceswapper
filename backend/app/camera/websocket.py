"""WebSocket endpoint accepting browser-captured JPEGs and returning processed frames."""
from __future__ import annotations

import asyncio
import hmac
import json
import os

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()
MAX_FRAME_BYTES = 8 * 1024 * 1024


@router.websocket("/ws/stream")
async def camera_stream(websocket: WebSocket) -> None:
    state = websocket.app.state.services
    expected_token = os.getenv("FRAME_DESKTOP_SESSION_TOKEN", "")
    if expected_token:
        allowed_origins = {None, "null", "file://", "http://127.0.0.1:5173", "http://localhost:5173"}
        origin = websocket.headers.get("origin")
        if origin not in allowed_origins:
            await websocket.close(code=1008, reason="Untrusted application origin.")
            return
    supplied_protocols = websocket.scope.get("subprotocols", [])
    if expected_token and not any(
        hmac.compare_digest(expected_token, protocol) for protocol in supplied_protocols
    ):
        await websocket.close(code=1008, reason="Local desktop session token is missing.")
        return
    if not state.camera.status()["active"]:
        await websocket.close(code=1008, reason="Start the camera from the application first.")
        return
    await websocket.accept(subprotocol="frame-v1" if "frame-v1" in supplied_protocols else None)
    state.camera.set_connected(True)
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            payload = message.get("bytes")
            if payload is None:
                text = message.get("text")
                if text is not None and len(text) <= 8_192:
                    try:
                        acknowledgement = json.loads(text)
                    except json.JSONDecodeError:
                        acknowledgement = None
                    if (
                        isinstance(acknowledgement, dict)
                        and acknowledgement.get("type") == "frame_diagnostics_displayed"
                        and isinstance(acknowledgement.get("trace_id"), str)
                    ):
                        # The renderer sends this only for a sampled diagnostic
                        # frame after createImageBitmap/canvas drawing. It closes
                        # the last observable boundary without changing preview UI.
                        state.frame_processor.record_frontend_diagnostics(acknowledgement)
                        continue
                await websocket.send_json({"type": "error", "message": "Expected a binary JPEG camera frame."})
                continue
            if len(payload) > MAX_FRAME_BYTES:
                await websocket.send_json({"type": "error", "message": "Camera frame exceeds the 8 MB limit."})
                continue
            try:
                frame, stats = await asyncio.to_thread(state.frame_processor.process_jpeg, payload)
                await websocket.send_json(stats)
                await websocket.send_bytes(frame)
            except (ValueError, RuntimeError) as exc:
                await websocket.send_json({"type": "error", "message": str(exc)})
    except WebSocketDisconnect:
        pass
    finally:
        state.camera.set_connected(False)
