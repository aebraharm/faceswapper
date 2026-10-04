"""WebSocket endpoint accepting browser-captured JPEGs and returning processed frames."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()
MAX_FRAME_BYTES = 8 * 1024 * 1024


@router.websocket("/ws/stream")
async def camera_stream(websocket: WebSocket) -> None:
    state = websocket.app.state.services
    if not state.camera.status()["active"]:
        await websocket.close(code=1008, reason="Start the camera from the application first.")
        return
    await websocket.accept()
    state.camera.set_connected(True)
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            payload = message.get("bytes")
            if payload is None:
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
