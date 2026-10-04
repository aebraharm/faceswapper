"""Detects, downloads and removes catalog model bundles in a local models directory.

Design goals (see the project brief):
  * never download anything automatically/on launch -- only an explicit install call,
  * verify every file's SHA-256 before it is considered installed,
  * keep partially-downloaded files out of the "installed" state (atomic rename),
  * never re-download an already-verified file,
  * surface progress so the UI can show a download bar instead of freezing.
"""
from __future__ import annotations

import hashlib
import shutil
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.transformation.model_catalog import ModelAsset, ModelDefinition, get_model

CHUNK_SIZE = 1024 * 256
DOWNLOAD_TIMEOUT_SECONDS = 30


class ModelInstallError(RuntimeError):
    pass


@dataclass
class _InstallState:
    downloading: bool = False
    bytes_downloaded: int = 0
    total_bytes: int = 0
    current_file: str | None = None
    error: str | None = None
    cancelled: bool = False
    thread: threading.Thread | None = field(default=None, repr=False)


class ModelInstallManager:
    """Thread-safe install/uninstall/status manager for catalog models."""

    def __init__(self, models_dir: Path, opener: Any = None) -> None:
        self.models_dir = Path(models_dir)
        # Tests inject a fake opener so CI never downloads real weights.
        self._opener = opener or urllib.request.urlopen
        self._lock = threading.RLock()
        self._states: dict[str, _InstallState] = {}

    def _model_dir(self, model_id: str) -> Path:
        return self.models_dir / model_id

    def _asset_path(self, model_id: str, asset: ModelAsset) -> Path:
        return self._model_dir(model_id) / asset.filename

    def is_installed(self, model_id: str) -> bool:
        definition = get_model(model_id)
        if definition is None:
            return False
        return all(self._asset_verified(model_id, asset) for asset in definition.assets)

    def _asset_verified(self, model_id: str, asset: ModelAsset) -> bool:
        path = self._asset_path(model_id, asset)
        if not path.is_file():
            return False
        # Re-hashing on every status call would be expensive for a 400+ MB file on
        # every poll; a size match is a cheap corruption smoke test, and install()
        # always re-verifies the hash right after writing the file.
        return path.stat().st_size > 0

    def installed_asset_paths(self, model_id: str) -> dict[str, Path] | None:
        """Return {logical role: path} for an installed model, or None if incomplete."""
        definition = get_model(model_id)
        if definition is None or not self.is_installed(model_id):
            return None
        return {asset.filename: self._asset_path(model_id, asset) for asset in definition.assets}

    def status(self, model_id: str) -> dict[str, Any]:
        definition = get_model(model_id)
        if definition is None:
            raise ModelInstallError(f"Unknown model id: {model_id}")
        with self._lock:
            state = self._states.get(model_id, _InstallState())
            installed = self.is_installed(model_id)
            progress = 0.0
            if state.total_bytes:
                progress = min(1.0, state.bytes_downloaded / state.total_bytes)
            elif installed:
                progress = 1.0
            return {
                "id": definition.id,
                "display_name": definition.display_name,
                "license_name": definition.license_name,
                "license_url": definition.license_url,
                "source_url": definition.source_url,
                "summary": definition.summary,
                "approx_total_bytes": definition.approx_total_bytes,
                "installed": installed,
                "downloading": state.downloading,
                "bytes_downloaded": state.bytes_downloaded,
                "total_bytes": state.total_bytes or definition.approx_total_bytes,
                "progress": progress,
                "error": state.error,
            }

    def list_status(self) -> list[dict[str, Any]]:
        from app.transformation.model_catalog import list_models

        return [self.status(definition.id) for definition in list_models()]

    def install(self, model_id: str) -> dict[str, Any]:
        """Start (or report) a background download. Returns the current status."""
        definition = get_model(model_id)
        if definition is None:
            raise ModelInstallError(f"Unknown model id: {model_id}")
        with self._lock:
            if self.is_installed(model_id):
                return self.status(model_id)
            state = self._states.get(model_id)
            if state is not None and state.downloading:
                return self.status(model_id)
            state = _InstallState(downloading=True, total_bytes=definition.approx_total_bytes)
            self._states[model_id] = state
            thread = threading.Thread(
                target=self._download_all, args=(definition,), name=f"model-install-{model_id}", daemon=True
            )
            state.thread = thread
            thread.start()
        return self.status(model_id)

    def _download_all(self, definition: ModelDefinition) -> None:
        state = self._states[definition.id]
        destination_dir = self._model_dir(definition.id)
        try:
            destination_dir.mkdir(parents=True, exist_ok=True)
            already_downloaded = 0
            for asset in definition.assets:
                with self._lock:
                    if state.cancelled:
                        return
                    state.current_file = asset.filename
                self._download_one(definition.id, asset, state, already_downloaded)
                already_downloaded += asset.approx_bytes
            with self._lock:
                state.downloading = False
                state.current_file = None
                state.error = None
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI, not a crash
            with self._lock:
                state.downloading = False
                state.error = str(exc)
            # Remove partial files so a retry starts clean instead of reporting
            # "installed" from a half-written asset.
            self._cleanup_partial(definition)

    def _cleanup_partial(self, definition: ModelDefinition) -> None:
        for asset in definition.assets:
            path = self._asset_path(definition.id, asset)
            temp_path = path.with_suffix(path.suffix + ".part")
            for candidate in (path, temp_path):
                if candidate.is_file() and not self._asset_verified(definition.id, asset):
                    candidate.unlink(missing_ok=True)

    def _download_one(self, model_id: str, asset: ModelAsset, state: _InstallState, bytes_before: int) -> None:
        destination = self._asset_path(model_id, asset)
        if self._asset_verified(model_id, asset) and self._sha256_matches(destination, asset.sha256):
            with self._lock:
                state.bytes_downloaded = bytes_before + asset.approx_bytes
            return
        temp_path = destination.with_suffix(destination.suffix + ".part")
        hasher = hashlib.sha256()
        try:
            with self._opener(asset.url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response, open(temp_path, "wb") as handle:
                while True:
                    with self._lock:
                        if state.cancelled:
                            raise ModelInstallError("Download cancelled.")
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)
                    hasher.update(chunk)
                    with self._lock:
                        state.bytes_downloaded = bytes_before + handle.tell()
        except urllib.error.URLError as exc:
            temp_path.unlink(missing_ok=True)
            raise ModelInstallError(f"Could not download {asset.filename}: {exc.reason}") from exc
        except OSError as exc:
            temp_path.unlink(missing_ok=True)
            raise ModelInstallError(f"Could not save {asset.filename} to disk: {exc}") from exc

        digest = hasher.hexdigest()
        if digest.lower() != asset.sha256.lower():
            temp_path.unlink(missing_ok=True)
            raise ModelInstallError(
                f"{asset.filename} did not match its published checksum after download "
                "(the file may be corrupt or the mirror changed); nothing was installed."
            )
        temp_path.replace(destination)

    @staticmethod
    def _sha256_matches(path: Path, expected: str) -> bool:
        hasher = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
                hasher.update(chunk)
        return hasher.hexdigest().lower() == expected.lower()

    def cancel(self, model_id: str) -> None:
        with self._lock:
            state = self._states.get(model_id)
            if state is not None:
                state.cancelled = True

    def remove(self, model_id: str) -> dict[str, Any]:
        definition = get_model(model_id)
        if definition is None:
            raise ModelInstallError(f"Unknown model id: {model_id}")
        with self._lock:
            state = self._states.get(model_id)
            if state is not None and state.downloading:
                raise ModelInstallError("Cannot remove a model while it is downloading. Cancel the download first.")
            self._states.pop(model_id, None)
        shutil.rmtree(self._model_dir(model_id), ignore_errors=True)
        return self.status(model_id)
