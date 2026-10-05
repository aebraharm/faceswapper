"""Dependency container for API routes and WebSocket frame processing."""
from __future__ import annotations

from dataclasses import dataclass

from app.alignment.face_aligner import FaceAligner
from app.blending.compositor import FaceCompositor
from app.camera.frame_processor import FrameProcessor
from app.camera.session import CameraSession
from app.config import Settings
from app.detection.analyzer import FaceAnalyzer
from app.detection.face_detector import OpenCVFaceDetector
from app.landmarks.landmark_detector import MediaPipeLandmarkDetector
from app.source_faces import SourceFaceStore
from app.transformation.manager import TransformerManager
from app.transformation.model_manager import ModelInstallManager


@dataclass
class Services:
    settings: Settings
    analyzer: FaceAnalyzer
    aligner: FaceAligner
    compositor: FaceCompositor
    transformer: TransformerManager
    source_faces: SourceFaceStore
    camera: CameraSession
    frame_processor: FrameProcessor
    model_store: ModelInstallManager


def create_services(settings: Settings | None = None) -> Services:
    config = settings or Settings.from_environment()
    landmarks = MediaPipeLandmarkDetector(
        max_num_faces=4, task_model_path=config.resolved_model_path(config.face_landmarker_task)
    )
    analyzer = FaceAnalyzer(OpenCVFaceDetector(), landmarks)
    aligner = FaceAligner()
    compositor = FaceCompositor()
    model_store = ModelInstallManager(config.resolved_models_dir())
    transformer = TransformerManager(config, model_store)
    source_faces = SourceFaceStore(analyzer, aligner, transformer)
    camera = CameraSession(config.default_processing_resolution)
    processor = FrameProcessor(
        analyzer,
        aligner,
        compositor,
        transformer,
        source_faces,
        camera,
        diagnostics_enabled=config.frame_diagnostics,
        diagnostics_interval_ms=config.frame_diagnostics_interval_ms,
    )
    return Services(config, analyzer, aligner, compositor, transformer, source_faces, camera, processor, model_store)
