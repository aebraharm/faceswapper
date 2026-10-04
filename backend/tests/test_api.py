from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.config import Settings
from app.main import create_app
from app.detection.types import BoundingBox, FaceObservation
from conftest import png_bytes


def make_client():
    settings = Settings(source_encoder_model=None, face_transformer_model=None)
    return TestClient(create_app(settings))


def test_health_and_browser_camera_devices_contract():
    with make_client() as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        devices = client.get("/camera/devices").json()
        assert devices["source"] == "browser"
        assert devices["requires_browser_permission"] is True


def test_camera_lifecycle_and_settings_validation():
    with make_client() as client:
        started = client.post("/camera/start", json={"device_id": "test-camera"})
        assert started.status_code == 200
        assert client.get("/camera/status").json()["active"] is True
        bad_settings = client.post("/camera/settings", json={"intensity": 2.0})
        assert bad_settings.status_code == 422
        stopped = client.post("/camera/stop")
        assert stopped.status_code == 200
        assert stopped.json()["active"] is False


def test_source_upload_rejects_image_without_a_face():
    with make_client() as client:
        response = client.post(
            "/source-face/upload",
            content=png_bytes(),
            headers={"Content-Type": "image/png"},
        )
        assert response.status_code == 422
        assert "No face" in response.json()["detail"]


def test_source_upload_receives_decodes_and_returns_valid_face_status():
    with make_client() as client:
        seen = {}
        face = FaceObservation(BoundingBox(40, 32, 72, 88))

        def analyze(rgb):
            seen["shape"] = rgb.shape
            seen["dtype"] = str(rgb.dtype)
            seen["first_pixel"] = tuple(int(value) for value in rgb[0, 0])
            return [face]

        client.app.state.services.analyzer.analyze = analyze
        response = client.post(
            "/source-face/upload",
            content=png_bytes(color=(12, 34, 56)),
            headers={"Content-Type": "image/png"},
        )

        assert response.status_code == 200
        assert seen == {"shape": (160, 160, 3), "dtype": "uint8", "first_pixel": (12, 34, 56)}
        body = response.json()
        assert body["uploaded"] is True
        assert body["face_count"] == 1
        assert body["faces"][0]["x"] == 40
        assert body["selected_face_index"] == 0
        assert body["ready"] is True
        assert body["width"] == 160
        assert body["height"] == 160
        assert body["storage"] == "memory only"
        assert body["format"] == "PNG"


def test_source_upload_rejects_corrupted_image_with_clear_error():
    with make_client() as client:
        response = client.post(
            "/source-face/upload",
            content=b"not a decodable image",
            headers={"Content-Type": "image/png"},
        )
        assert response.status_code == 422
        assert "could not be decoded" in response.json()["detail"]


def test_source_upload_rejects_mime_mismatch():
    with make_client() as client:
        response = client.post(
            "/source-face/upload",
            content=png_bytes(),
            headers={"Content-Type": "image/jpeg"},
        )
        assert response.status_code == 415


def test_source_upload_enforces_streaming_size_limit():
    with TestClient(create_app(Settings(max_upload_bytes=32))) as client:
        response = client.post(
            "/source-face/upload",
            content=b"x" * 33,
            headers={"Content-Type": "image/png"},
        )
        assert response.status_code == 413


def test_missing_model_returns_actionable_error_and_status():
    with make_client() as client:
        response = client.post("/transformer/load", json={"provider": "auto"})
        assert response.status_code == 503
        assert "models/README.md" in response.json()["detail"]
        status = client.get("/transformer/status").json()
        assert status["loaded"] is False
        assert status["error"]


def test_source_face_api_requires_explicit_selection_for_multiple_faces():
    with make_client() as client:
        faces = [
            FaceObservation(BoundingBox(20, 20, 45, 50)),
            FaceObservation(BoundingBox(90, 24, 48, 53)),
        ]
        client.app.state.services.analyzer.analyze = lambda image: faces
        response = client.post(
            "/source-face/upload",
            content=png_bytes(),
            headers={"Content-Type": "image/png"},
        )
        assert response.status_code == 200
        assert response.json()["face_count"] == 2
        assert response.json()["selected_face_index"] is None
        assert response.json()["ready"] is False

        selected = client.post("/source-face/select", json={"face_index": 1})
        assert selected.status_code == 200
        assert selected.json()["selected_face_index"] == 1
        assert selected.json()["ready"] is True


def test_transformation_cannot_be_enabled_without_source_and_model():
    with make_client() as client:
        response = client.post("/camera/settings", json={"transform_enabled": True})
        assert response.status_code == 409
        assert "Load a compatible" in response.json()["detail"]


def test_camera_websocket_returns_binary_preview_and_performance_stats():
    import cv2
    import numpy as np

    with make_client() as client:
        client.post("/camera/start", json={})
        ok, frame = cv2.imencode(".jpg", np.zeros((120, 160, 3), dtype=np.uint8))
        assert ok
        with client.websocket_connect("/ws/stream") as websocket:
            websocket.send_bytes(frame.tobytes())
            stats = websocket.receive_json()
            processed = websocket.receive_bytes()
            assert stats["type"] == "stats"
            assert stats["camera_resolution"] == "160 × 120"
            assert stats["transformation_active"] is False
            assert processed[:2] == b"\xff\xd8"
        client.post("/camera/stop")


def test_desktop_session_token_protects_http_routes(monkeypatch):
    with make_client() as client:
        monkeypatch.setenv("FRAME_DESKTOP_SESSION_TOKEN", "session-secret")
        assert client.get("/health").status_code == 404
        assert client.get("/health", headers={"X-Frame-Session": "session-secret"}).status_code == 200


def test_packaged_renderer_can_preflight_authenticated_requests(monkeypatch):
    with make_client() as client:
        monkeypatch.setenv("FRAME_DESKTOP_SESSION_TOKEN", "session-secret")
        response = client.options(
            "/health",
            headers={
                "Origin": "null",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "x-frame-session",
            },
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "null"


def test_packaged_renderer_can_preflight_and_post_an_authenticated_image_upload(monkeypatch):
    with make_client() as client:
        monkeypatch.setenv("FRAME_DESKTOP_SESSION_TOKEN", "session-secret")
        for origin in ("null", "file://"):
            preflight = client.options(
                "/source-face/upload",
                headers={
                    "Origin": origin,
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type,x-frame-session",
                },
            )
            assert preflight.status_code == 200
            assert preflight.headers["access-control-allow-origin"] == origin
            assert "POST" in preflight.headers["access-control-allow-methods"]

        client.app.state.services.analyzer.analyze = lambda image: []
        response = client.post(
            "/source-face/upload",
            content=png_bytes(),
            headers={
                "Origin": "null",
                "Content-Type": "image/png",
                "X-Frame-Session": "session-secret",
            },
        )
        assert response.status_code == 422
        assert response.headers["access-control-allow-origin"] == "null"
        assert "No face" in response.json()["detail"]


def test_desktop_websocket_requires_session_subprotocol(monkeypatch):
    with make_client() as client:
        monkeypatch.setenv("FRAME_DESKTOP_SESSION_TOKEN", "session-secret")
        client.post("/camera/start", json={}, headers={"X-Frame-Session": "session-secret"})
        with pytest.raises(WebSocketDisconnect) as untrusted_origin:
            with client.websocket_connect(
                "/ws/stream",
                headers={"origin": "https://untrusted.example"},
                subprotocols=["frame-v1", "session-secret"],
            ):
                pass
        assert untrusted_origin.value.code == 1008

        with pytest.raises(WebSocketDisconnect) as disconnected:
            with client.websocket_connect(
                "/ws/stream",
                headers={"origin": "file://"},
                subprotocols=["frame-v1"],
            ):
                pass
        assert disconnected.value.code == 1008

        with client.websocket_connect(
            "/ws/stream",
            headers={"origin": "file://"},
            subprotocols=["frame-v1", "session-secret"],
        ) as websocket:
            assert websocket.accepted_subprotocol == "frame-v1"
        client.post("/camera/stop", headers={"X-Frame-Session": "session-secret"})


def test_desktop_shutdown_hook_requires_matching_session_and_shutdown_tokens(monkeypatch):
    with make_client() as client:
        requested = []
        client.app.state.request_desktop_shutdown = lambda: requested.append(True)
        monkeypatch.setenv("FRAME_DESKTOP_SESSION_TOKEN", "session-secret")
        monkeypatch.setenv("FRAME_DESKTOP_SHUTDOWN_TOKEN", "shutdown-secret")

        wrong_session = client.post(
            "/internal/shutdown",
            headers={"X-Frame-Session": "wrong", "X-Frame-Shutdown-Token": "shutdown-secret"},
        )
        assert wrong_session.status_code == 404
        wrong_shutdown = client.post(
            "/internal/shutdown",
            headers={"X-Frame-Session": "session-secret", "X-Frame-Shutdown-Token": "wrong"},
        )
        assert wrong_shutdown.status_code == 404
        accepted = client.post(
            "/internal/shutdown",
            headers={"X-Frame-Session": "session-secret", "X-Frame-Shutdown-Token": "shutdown-secret"},
        )
        assert accepted.status_code == 200
        assert requested == [True]
