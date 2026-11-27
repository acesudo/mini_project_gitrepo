"""
Module 10 — Face Detection & Image Quality Gate Tests.

Verifies:
1. Haar Cascade models load successfully.
2. Synthetic frontal portraits pass quality gates.
3. Overexposed, underexposed, blurry, and malformed frames are accurately rejected.
4. Solid canvases produce zero false-positive detections.
"""

import cv2
import numpy as np

from module2.harr_classifier import detect_faces, face_cascade
from module3.face_quality import validate_face_image
from module10.synthetic_data import create_synthetic_face, apply_lighting


def test_face_cascade_loaded_successfully():
    """Haar Cascade classifier should be properly loaded and non-empty."""
    assert face_cascade is not None
    assert not face_cascade.empty()


def test_face_quality_acceptable_synthetic_face():
    """A clean synthetic portrait (160x160x3) should pass quality checks."""
    face = create_synthetic_face(seed=101, size=(160, 160))
    # Convert BGR to RGB as required by validate_face_image
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    is_valid, msg = validate_face_image(face_rgb)
    assert is_valid is True
    assert "acceptable" in msg.lower()


def test_face_quality_rejects_dark_image():
    """Severely underexposed image (< 40 mean intensity) must be rejected."""
    face = create_synthetic_face(seed=102, size=(160, 160))
    dark_face = apply_lighting(face, factor=0.1)
    dark_face_rgb = cv2.cvtColor(dark_face, cv2.COLOR_BGR2RGB)

    is_valid, msg = validate_face_image(dark_face_rgb)
    assert is_valid is False
    assert "dark" in msg.lower()


def test_face_quality_rejects_bright_image():
    """Overexposed image (> 220 mean intensity) must be rejected."""
    bright_face = np.full((160, 160, 3), 240, dtype=np.uint8)
    is_valid, msg = validate_face_image(bright_face)
    assert is_valid is False
    assert "bright" in msg.lower()


def test_face_quality_rejects_blurry_image():
    """Heavy Gaussian blur (Laplacian variance < 50) must be rejected."""
    face = create_synthetic_face(seed=104, size=(160, 160))
    # Apply aggressive 21x21 Gaussian blur
    blurry_face = cv2.GaussianBlur(face, (21, 21), 0)
    blurry_face_rgb = cv2.cvtColor(blurry_face, cv2.COLOR_BGR2RGB)

    is_valid, msg = validate_face_image(blurry_face_rgb)
    assert is_valid is False
    assert "blurry" in msg.lower()


def test_face_quality_rejects_invalid_dimensions():
    """Images with dimensions other than 160x160x3 must be rejected."""
    wrong_dims = np.zeros((120, 120, 3), dtype=np.uint8)
    is_valid, msg = validate_face_image(wrong_dims)
    assert is_valid is False
    assert "160x160x3" in msg

    empty_img = np.array([])
    is_valid, msg = validate_face_image(empty_img)
    assert is_valid is False


def test_haar_detector_empty_or_solid_canvas():
    """A flat black or white canvas should produce zero false face detections."""
    black_canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    faces_black = detect_faces(black_canvas)
    assert len(faces_black) == 0

    white_canvas = np.full((480, 640, 3), 255, dtype=np.uint8)
    faces_white = detect_faces(white_canvas)
    assert len(faces_white) == 0
