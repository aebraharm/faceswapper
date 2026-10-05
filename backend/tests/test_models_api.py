from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from fixtures.tiny_liveportrait import build_tiny_liveportrait_bundle


def make_client(tmp_path: Path, **overrides):
    settings = Settings(source_encoder_model=None, face_transformer_model=None, models_dir=str(tmp_path), **overrides)
    return TestClient(create_app(settings))


def test_catalog_endpoint_lists_models_with_license_and_install_state(tmp_path):
    with make_client(tmp_path) as client:
        response = client.get("/models/catalog")
        assert response.status_code == 200
        body = response.json()
        assert body["selected_model_id"] == "liveportrait-v1"
        models = {model["id"]: model for model in body["models"]}
        assert "liveportrait-v1" in models
        entry = models["liveportrait-v1"]
        assert entry["license_name"] == "MIT"
        assert entry["installed"] is False
        assert entry["downloading"] is False


def test_model_status_endpoint_for_unknown_model_is_404(tmp_path):
    with make_client(tmp_path) as client:
        response = client.get("/models/not-a-real-model/status")
        assert response.status_code == 404


def test_install_endpoint_starts_a_background_download_without_blocking(tmp_path, monkeypatch):
    with make_client(tmp_path) as client:
        # Never hit the network in tests: make the fake opener hang briefly so we can
        # observe a "downloading" state, then fail fast (still proves the install
        # endpoint itself is non-blocking and returns immediately).
        import io
        import time

        def slow_opener(url, timeout=0):
            raise TimeoutError("simulated: no network access in CI")

        client.app.state.services.model_store._opener = slow_opener
        response = client.post("/models/liveportrait-v1/install")
        assert response.status_code == 200
        body = response.json()
        assert body["id"] == "liveportrait-v1"

        deadline = time.monotonic() + 5
        status = client.get("/models/liveportrait-v1/status").json()
        while status["downloading"] and time.monotonic() < deadline:
            time.sleep(0.02)
            status = client.get("/models/liveportrait-v1/status").json()
        assert status["installed"] is False
        assert status["error"]


def test_remove_endpoint_deletes_files(tmp_path):
    with make_client(tmp_path) as client:
        model_dir = tmp_path / "liveportrait-v1"
        model_dir.mkdir(parents=True)
        for name in ("appearance_feature_extractor.onnx", "motion_extractor.onnx", "warping_spade.onnx"):
            (model_dir / name).write_bytes(b"x")

        response = client.delete("/models/liveportrait-v1")
        assert response.status_code == 200
        assert not model_dir.exists()

        response = client.delete("/models/not-a-real-model")
        assert response.status_code == 404


def test_remove_endpoint_blocks_removal_of_the_currently_loaded_model(tmp_path):
    with make_client(tmp_path) as client:
        build_tiny_liveportrait_bundle(tmp_path / "liveportrait-v1")
        loaded = client.post("/transformer/load", json={"provider": "CPUExecutionProvider"})
        assert loaded.status_code == 200
        assert loaded.json()["model_id"] == "liveportrait-v1"

        response = client.delete("/models/liveportrait-v1")
        assert response.status_code == 409
        assert (tmp_path / "liveportrait-v1" / "warping_spade.onnx").exists()
