from __future__ import annotations

import hashlib
import io
from pathlib import Path
from urllib.error import URLError

import pytest

from app.transformation.model_catalog import CATALOG, get_model, list_models
from app.transformation.model_manager import ModelInstallError, ModelInstallManager


def test_catalog_entries_are_well_formed():
    models = list_models()
    assert models, "catalog should not be empty"
    for model in models:
        assert get_model(model.id) is model
        assert model.license_name and model.license_url.startswith("https://")
        assert model.assets, f"{model.id} has no assets"
        for asset in model.assets:
            assert len(asset.sha256) == 64
            assert asset.url.startswith("https://")
            assert asset.approx_bytes > 0


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _fake_opener_for(contents_by_url: dict[str, bytes], raise_for: set[str] | None = None):
    raise_for = raise_for or set()

    def opener(url, timeout=0):  # noqa: ARG001 - signature mirrors urllib.request.urlopen
        if url in raise_for:
            raise URLError("simulated network failure")
        return _FakeResponse(contents_by_url[url])

    return opener


def _wait_for_install(manager: ModelInstallManager, model_id: str, timeout: float = 5.0) -> dict:
    import time

    deadline = time.monotonic() + timeout
    status = manager.status(model_id)
    while status["downloading"] and time.monotonic() < deadline:
        time.sleep(0.02)
        status = manager.status(model_id)
    return status


def test_install_downloads_verifies_and_marks_model_installed(tmp_path: Path):
    model = CATALOG["liveportrait-v1"]
    contents = {asset.url: f"content-for-{asset.filename}".encode() for asset in model.assets}
    # Rebuild the catalog entry's checksums against our fake content so install() succeeds
    # without needing the real ~537 MB download in CI.
    import dataclasses

    patched_assets = tuple(
        dataclasses.replace(asset, sha256=hashlib.sha256(contents[asset.url]).hexdigest())
        for asset in model.assets
    )
    patched_model = dataclasses.replace(model, assets=patched_assets)
    import app.transformation.model_catalog as catalog_module

    original = catalog_module.CATALOG[model.id]
    catalog_module.CATALOG[model.id] = patched_model
    try:
        manager = ModelInstallManager(tmp_path, opener=_fake_opener_for(contents))
        assert manager.is_installed(model.id) is False
        manager.install(model.id)
        status = _wait_for_install(manager, model.id)
        assert status["error"] is None
        assert status["installed"] is True
        assert manager.is_installed(model.id) is True
        paths = manager.installed_asset_paths(model.id)
        assert paths is not None and set(paths) == {asset.filename for asset in model.assets}
    finally:
        catalog_module.CATALOG[model.id] = original


def test_install_rejects_and_cleans_up_a_checksum_mismatch(tmp_path: Path):
    model = CATALOG["liveportrait-v1"]
    contents = {asset.url: b"not the expected bytes" for asset in model.assets}
    manager = ModelInstallManager(tmp_path, opener=_fake_opener_for(contents))
    manager.install(model.id)
    status = _wait_for_install(manager, model.id)
    assert status["installed"] is False
    assert status["error"] and "checksum" in status["error"].lower()
    # No partial/corrupt file should remain on disk.
    assert not any((tmp_path / model.id).glob("*"))


def test_install_surfaces_network_errors_without_crashing(tmp_path: Path):
    model = CATALOG["liveportrait-v1"]
    first_url = model.assets[0].url
    contents = {asset.url: b"x" for asset in model.assets}
    manager = ModelInstallManager(tmp_path, opener=_fake_opener_for(contents, raise_for={first_url}))
    manager.install(model.id)
    status = _wait_for_install(manager, model.id)
    assert status["installed"] is False
    assert status["error"]


def test_remove_deletes_installed_files_and_unknown_model_errors(tmp_path: Path):
    model = CATALOG["liveportrait-v1"]
    (tmp_path / model.id).mkdir(parents=True)
    for asset in model.assets:
        (tmp_path / model.id / asset.filename).write_bytes(b"x")
    manager = ModelInstallManager(tmp_path)
    manager.remove(model.id)
    assert not (tmp_path / model.id).exists()

    with pytest.raises(ModelInstallError):
        manager.status("not-a-real-model")
    with pytest.raises(ModelInstallError):
        manager.install("not-a-real-model")
    with pytest.raises(ModelInstallError):
        manager.remove("not-a-real-model")


def test_status_reports_unverified_as_not_installed_until_asset_present(tmp_path: Path):
    model = CATALOG["liveportrait-v1"]
    manager = ModelInstallManager(tmp_path)
    status = manager.status(model.id)
    assert status["installed"] is False
    assert status["downloading"] is False
    assert status["progress"] == 0.0
    assert status["approx_total_bytes"] == model.approx_total_bytes
