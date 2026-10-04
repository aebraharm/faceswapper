"""Per-frame detector → tracker → aligner → model → mask/blend → JPEG pipeline."""
from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from app.alignment.face_aligner import FaceAligner
from app.blending.compositor import FaceCompositor
from app.camera.session import CameraSession
from app.camera.target_tracker import TargetFaceTracker
from app.detection.analyzer import FaceAnalyzer
from app.transformation.manager import TransformerManager


class FrameProcessor:
    def __init__(
        self,
        analyzer: FaceAnalyzer,
        aligner: FaceAligner,
        compositor: FaceCompositor,
        transformer: TransformerManager,
        source_faces: Any,
        camera: CameraSession,
    ) -> None:
        self.analyzer = analyzer
        self.aligner = aligner
        self.compositor = compositor
        self.transformer = transformer
        self.source_faces = source_faces
        self.camera = camera
        self._tracker = TargetFaceTracker()

    def process_jpeg(self, payload: bytes) -> tuple[bytes, dict[str, Any]]:
        started = time.perf_counter()
        encoded = np.frombuffer(payload, dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Camera frame is not a valid JPEG image.")
        height, width = bgr.shape[:2]
        if width < 2 or height < 2:
            raise ValueError("Camera frame dimensions are invalid.")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        settings = self.camera.settings()
        max_dimension = settings["processing_resolution"]
        scale = min(1.0, max_dimension / float(max(width, height)))
        if scale < 1.0:
            work = cv2.resize(rgb, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
        else:
            work = rgb.copy()

        observations = self.analyzer.analyze(work)
        selected_preference = self.camera.status()["selected_target_index"]
        selected_index, target = self._tracker.select(observations, selected_preference)
        # If there is a single face the tracker chooses it automatically; multiple faces
        # remain untouched until the user selects one explicitly.
        self.camera.update_faces([face.as_dict() for face in observations], selected_index)
        output = work
        model_status = self.transformer.status()
        source_status = self.source_faces.status()
        transform_active = bool(
            settings["transform_enabled"]
            and model_status["loaded"]
            and source_status["model_ready"]
            and target is not None
        )
        if transform_active and target is not None:
            representation = self.source_faces.representation()
            if representation is not None:
                aligned = self.aligner.align(work, target, output_size=256)
                transformed = self.transformer.transform(representation, aligned.image)
                output = self.compositor.blend(work, transformed, aligned, settings["intensity"])
            else:
                transform_active = False

        # Highlight all detected faces; only the selected target is marked in green.
        annotated = output.copy()
        for i, face in enumerate(observations):
            x, y, box_w, box_h = face.bbox
            color = (63, 217, 163) if i == selected_index else (246, 178, 93)
            thickness = max(1, int(round(min(work.shape[:2]) / 360)))
            cv2.rectangle(annotated, (x, y), (x + box_w, y + box_h), color, thickness, cv2.LINE_AA)
        result = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        ok, jpeg = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            raise RuntimeError("Could not encode the processed camera frame.")
        latency_ms = (time.perf_counter() - started) * 1000.0
        self.camera.record_frame(latency_ms, width, height)
        camera_status = self.camera.status()
        stats = {
            "type": "stats",
            "fps": camera_status["fps"],
            "latency_ms": round(latency_ms, 1),
            "camera_resolution": camera_status["camera_resolution"],
            "processing_resolution": max(work.shape[1], work.shape[0]),
            "face_count": len(observations),
            "faces": [face.as_dict() for face in observations],
            "selected_target": selected_index,
            "transformation_active": transform_active,
            "model_status": model_status,
            "device": model_status["device"],
        }
        return jpeg.tobytes(), stats
