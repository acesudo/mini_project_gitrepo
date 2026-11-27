"""
Module 10 — Procedural Synthetic Face & Environment Generator.

Generates reproducible facial test images, varied illumination,
in-plane rotations, multi-subject classroom frames, and noise patterns
for automated robustness benchmarking without camera hardware.
"""

from typing import List, Tuple
import cv2
import numpy as np


def create_synthetic_face(
    seed: int = 42,
    size: Tuple[int, int] = (160, 160),
    skin_tone: Tuple[int, int, int] = (180, 200, 230),  # BGR
) -> np.ndarray:
    """
    Generate a deterministic synthetic portrait with eye, nose, and mouth features.
    """
    np.random.seed(seed)
    h, w = size
    # Base background
    canvas = np.full((h, w, 3), 40, dtype=np.uint8)

    # Face oval
    center = (w // 2, h // 2)
    axes = (w // 3, int(h * 0.42))
    cv2.ellipse(canvas, center, axes, 0, 0, 360, skin_tone, -1)

    # Eyes
    eye_y = int(h * 0.42)
    left_eye_x = int(w * 0.38)
    right_eye_x = int(w * 0.62)
    cv2.circle(canvas, (left_eye_x, eye_y), int(w * 0.05), (40, 30, 20), -1)
    cv2.circle(canvas, (right_eye_x, eye_y), int(w * 0.05), (40, 30, 20), -1)

    # Nose
    nose_pts = np.array([
        [w // 2, int(h * 0.48)],
        [int(w * 0.47), int(h * 0.58)],
        [int(w * 0.53), int(h * 0.58)],
    ])
    cv2.fillPoly(canvas, [nose_pts], (150, 170, 200))

    # Mouth
    mouth_center = (w // 2, int(h * 0.70))
    mouth_axes = (int(w * 0.15), int(h * 0.06))
    cv2.ellipse(canvas, mouth_center, mouth_axes, 0, 0, 180, (60, 60, 160), -1)

    # Subtle textural variation
    noise = np.random.normal(0, 4, canvas.shape).astype(np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return canvas


def apply_lighting(img: np.ndarray, factor: float) -> np.ndarray:
    """
    Adjust brightness simulating dim lighting (< 1.0) or overexposure (> 1.0).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return result


def apply_rotation(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate image around center by angle_deg (simulates head tilt/roll).
    """
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(img, rot_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    return rotated


def create_multi_face_frame(
    faces: List[np.ndarray],
    positions: List[Tuple[int, int]],
    frame_size: Tuple[int, int] = (480, 640),
) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    """
    Place multiple face images into a classroom-like background frame.
    Returns:
        (frame: np.ndarray, bboxes: List[(x, y, w, h)])
    """
    fh, fw = frame_size
    frame = np.full((fh, fw, 3), 70, dtype=np.uint8)  # neutral background

    bboxes = []
    for face, (x, y) in zip(faces, positions):
        h, w = face.shape[:2]
        # Clip to bounds
        if y + h <= fh and x + w <= fw and x >= 0 and y >= 0:
            frame[y : y + h, x : x + w] = face
            bboxes.append((x, y, w, h))

    return frame, bboxes


def create_gaussian_noise(size: Tuple[int, int] = (160, 160)) -> np.ndarray:
    """
    Generate an adversarial non-face Gaussian noise image.
    """
    return np.random.randint(0, 256, (size[0], size[1], 3), dtype=np.uint8)
