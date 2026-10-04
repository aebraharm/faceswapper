"""Thread-safe status/settings for one browser-authorized camera stream."""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any


class CameraSession:
    ALLOWED_RESOLUTIONS = (320, 480, 640, 720)
    ALLOWED_PERFORMANCE_MODES = ("auto", "quality", "performance")
    # Below this rolling average FPS, "auto" mode starts skipping model inference
    # on alternating frames (holding the last composited result) instead of
    # falling further behind or crashing.
    _AUTO_SKIP_FPS_THRESHOLD = 6.0
    _AUTO_SKIP_MIN_SAMPLES = 8

    def __init__(self, default_resolution: int = 640) -> None:
        self._lock = threading.RLock()
        self.active = False
        self.connected = False
        self.device_id: str | None = None
        self.selected_target_index: int | None = None
        self.transform_enabled = False
        self.intensity = 0.85
        self.processing_resolution = default_resolution if default_resolution in self.ALLOWED_RESOLUTIONS else 640
        self.performance_mode = "auto"
        self.face_count = 0
        self.faces: list[dict[str, int | float | None]] = []
        self.selected_target: int | None = None
        self.frames_processed = 0
        self.fps = 0.0
        self.latency_ms = 0.0
        self.camera_resolution = "—"
        self._latencies: deque[float] = deque(maxlen=90)
        self._frame_times: deque[float] = deque(maxlen=31)
        self._inference_frame_counter = 0
        self._skipped_last = False

    def start(self, device_id: str | None = None) -> None:
        with self._lock:
            self.active = True
            self.connected = False
            self.device_id = device_id
            self.transform_enabled = False
            self.selected_target_index = None
            self.selected_target = None
            self.frames_processed = 0
            self.fps = 0.0
            self._latencies.clear()
            self._frame_times.clear()
            self._inference_frame_counter = 0
            self._skipped_last = False

    def stop(self) -> None:
        with self._lock:
            self.active = False
            self.connected = False
            self.transform_enabled = False
            self.selected_target_index = None
            self.selected_target = None
            self.faces = []
            self.face_count = 0
            self.fps = 0.0
            self.latency_ms = 0.0
            self.camera_resolution = "—"
            self._frame_times.clear()
            self._latencies.clear()

    def set_connected(self, connected: bool) -> None:
        with self._lock:
            self.connected = connected

    def configure(
        self,
        *,
        transform_enabled: bool | None = None,
        intensity: float | None = None,
        processing_resolution: int | None = None,
        performance_mode: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if transform_enabled is not None:
                self.transform_enabled = transform_enabled
            if intensity is not None:
                self.intensity = intensity
            if processing_resolution is not None:
                if processing_resolution not in self.ALLOWED_RESOLUTIONS:
                    raise ValueError(f"processing_resolution must be one of {self.ALLOWED_RESOLUTIONS}.")
                self.processing_resolution = processing_resolution
            if performance_mode is not None:
                if performance_mode not in self.ALLOWED_PERFORMANCE_MODES:
                    raise ValueError(f"performance_mode must be one of {self.ALLOWED_PERFORMANCE_MODES}.")
                self.performance_mode = performance_mode
            return self.settings()

    def should_run_inference(self) -> bool:
        """Adaptive frame-skip: decide whether this frame should run the (expensive)
        model transform, or reuse the last composited output to stay responsive.
        """
        with self._lock:
            self._inference_frame_counter += 1
            if self.performance_mode == "quality":
                self._skipped_last = False
                return True
            if self.performance_mode == "performance":
                skip = self._inference_frame_counter % 2 == 0
                self._skipped_last = skip
                return not skip
            # auto: only start skipping once we have enough samples to trust the
            # rolling FPS, and the stream is demonstrably struggling.
            if self.frames_processed >= self._AUTO_SKIP_MIN_SAMPLES and 0 < self.fps < self._AUTO_SKIP_FPS_THRESHOLD:
                skip = self._inference_frame_counter % 2 == 0
                self._skipped_last = skip
                return not skip
            self._skipped_last = False
            return True

    def select_target(self, index: int | None) -> None:
        with self._lock:
            if index is not None and (index < 0 or index >= 4):
                raise ValueError("Target face index must be between 0 and 3.")
            self.selected_target_index = index

    def update_faces(self, faces: list[dict[str, int | float | None]], selected_index: int | None) -> None:
        with self._lock:
            self.faces = faces
            self.face_count = len(faces)
            self.selected_target = selected_index

    def record_frame(self, latency_ms: float, width: int, height: int) -> None:
        now = time.perf_counter()
        with self._lock:
            self.frames_processed += 1
            self._latencies.append(latency_ms)
            self._frame_times.append(now)
            self.latency_ms = sum(self._latencies) / len(self._latencies)
            if len(self._frame_times) > 1:
                elapsed = self._frame_times[-1] - self._frame_times[0]
                self.fps = (len(self._frame_times) - 1) / elapsed if elapsed > 0 else 0.0
            self.camera_resolution = f"{width} × {height}"

    def settings(self) -> dict[str, Any]:
        with self._lock:
            return {
                "transform_enabled": self.transform_enabled,
                "intensity": self.intensity,
                "processing_resolution": self.processing_resolution,
                "performance_mode": self.performance_mode,
                "device_id": self.device_id,
            }

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active": self.active,
                "connected": self.connected,
                "device_id": self.device_id,
                "face_count": self.face_count,
                "faces": list(self.faces),
                "selected_target": self.selected_target,
                "selected_target_index": self.selected_target_index,
                "transform_enabled": self.transform_enabled,
                "frames_processed": self.frames_processed,
                "fps": round(self.fps, 1),
                "latency_ms": round(self.latency_ms, 1),
                "camera_resolution": self.camera_resolution,
                "processing_resolution": self.processing_resolution,
                "performance_mode": self.performance_mode,
                "skipped_last_frame": self._skipped_last,
            }
