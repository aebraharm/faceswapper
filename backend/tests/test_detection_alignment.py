from __future__ import annotations

import numpy as np

from app.alignment.face_aligner import FaceAligner
from app.detection.analyzer import FaceAnalyzer
from app.detection.face_detector import OpenCVFaceDetector
from app.detection.types import BoundingBox, FaceObservation


class Detector:
    def __init__(self, faces):
        self.faces = faces

    def detect(self, image):
        return self.faces


class Landmarks:
    available = True

    def __init__(self, faces):
        self.faces = faces

    def analyze(self, image):
        return self.faces


class Cascade:
    def detectMultiScale(self, image, **kwargs):
        return np.asarray([[12, 18, 36, 42]], dtype=np.int32)


def test_face_observation_normalizes_tuple_bounding_box():
    face = FaceObservation((1, 2, 30, 40))
    assert face.bbox == BoundingBox(1, 2, 30, 40)


def test_face_observation_normalizes_list_bounding_box():
    face = FaceObservation([1, 2, 30, 40])
    assert face.bbox == BoundingBox(1, 2, 30, 40)


def test_face_observation_normalizes_numpy_bounding_box():
    face = FaceObservation(np.asarray([1, 2, 30, 40], dtype=np.int32))
    assert face.bbox == BoundingBox(1, 2, 30, 40)


def test_face_observation_keeps_typed_bounding_box():
    bbox = BoundingBox(1, 2, 30, 40)
    assert FaceObservation(bbox).bbox is bbox


def test_opencv_detector_wraps_detected_boxes_in_bounding_boxes():
    detector = object.__new__(OpenCVFaceDetector)
    detector._cascade = Cascade()
    detector._scale_factor = 1.12
    detector._min_neighbors = 5

    faces = detector.detect(np.zeros((80, 80, 3), dtype=np.uint8))

    assert len(faces) == 1
    assert faces[0].bbox == BoundingBox(12, 18, 36, 42)
    assert isinstance(faces[0].bbox, BoundingBox)


def test_face_analyzer_uses_landmarks_and_falls_back():
    fallback_face = FaceObservation(BoundingBox(1, 2, 10, 12))
    landmark_face = FaceObservation(BoundingBox(3, 4, 20, 24), np.zeros((5, 2), dtype=np.float32))
    analyzer = FaceAnalyzer(Detector([fallback_face]), Landmarks([landmark_face]))
    result = analyzer.analyze(np.zeros((64, 64, 3), np.uint8))
    assert len(result) == 1 and result[0] is landmark_face

    empty_landmarks = FaceAnalyzer(Detector([fallback_face]), Landmarks([]))
    result = empty_landmarks.analyze(np.zeros((64, 64, 3), np.uint8))
    assert len(result) == 1 and result[0] is fallback_face


def test_alignment_fallback_and_landmark_paths_return_inverse_matrices():
    rgb = np.full((240, 300, 3), 120, dtype=np.uint8)
    box_face = FaceObservation(BoundingBox(80, 45, 100, 130))
    aligner = FaceAligner(output_size=128)
    result = aligner.align(rgb, box_face)
    assert result.image.shape == (128, 128, 3)
    assert result.forward_matrix.shape == (2, 3)
    assert result.inverse_matrix.shape == (2, 3)

    landmarks = np.array([[120, 100], [150, 100], [135, 120], [124, 140], [146, 140]], dtype=np.float32)
    aligned = aligner.align(rgb, FaceObservation(BoundingBox(100, 80, 70, 90), landmarks))
    assert aligned.image.shape == (128, 128, 3)
    assert np.isfinite(aligned.forward_matrix).all()


def test_target_tracker_keeps_selected_face_by_nearest_center_across_detection_reordering():
    from app.camera.target_tracker import TargetFaceTracker

    tracker = TargetFaceTracker(smoothing=1.0)
    left = FaceObservation(BoundingBox(10, 10, 30, 30))
    right = FaceObservation(BoundingBox(100, 10, 30, 30))
    first_index, _ = tracker.select([left, right], selected_index=0)
    # Detector order changes: track the face closest to the previous selected location.
    next_index, tracked = tracker.select([right, left], selected_index=0)
    assert first_index == 0
    assert next_index == 1
    assert tracked is not None and tracked.bbox.x == 10
