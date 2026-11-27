"""
Module 10 — Robustness & Environmental Variation Tests.

Tests:
1. Illumination variance (dim light, glare) and embedding stability.
2. In-plane pose tilt (affine rotation) stability.
3. Multi-subject frame generation, bounding, and separation.
4. Unknown impostor rejection against gallery (score < 0.70 threshold).
5. Adversarial noise rejection (Gaussian noise discrimination).
6. FAR/FRR metric calculation benchmarking.
"""

import numpy as np

from module4.face_embedding import FaceEmbeddingGenerator
from module5.face_matcher import FaceMatcher
from module10.synthetic_data import (
    create_synthetic_face,
    apply_lighting,
    apply_rotation,
    create_multi_face_frame,
    create_gaussian_noise,
)
from module10.test_bench import RobustnessBenchmark


def test_lighting_robustness_embedding_stability():
    """Embedding should remain stable under mild lighting shifts."""
    gen = FaceEmbeddingGenerator()
    face = create_synthetic_face(seed=55)

    base_emb = gen.generate_embedding(face)

    # 1. Dim lighting (factor = 0.75)
    dim_face = apply_lighting(face, factor=0.75)
    dim_emb = gen.generate_embedding(dim_face)
    sim_dim = float(np.dot(base_emb, dim_emb))
    assert sim_dim > 0.85, f"Dim lighting similarity {sim_dim} fell below 0.85"

    # 2. Bright lighting (factor = 1.25)
    bright_face = apply_lighting(face, factor=1.25)
    bright_emb = gen.generate_embedding(bright_face)
    sim_bright = float(np.dot(base_emb, bright_emb))
    assert sim_bright > 0.85, f"Bright lighting similarity {sim_bright} fell below 0.85"


def test_pose_rotation_embedding_stability():
    """Small in-plane head tilt (±10 degrees) should retain high cosine similarity."""
    gen = FaceEmbeddingGenerator()
    face = create_synthetic_face(seed=77)
    base_emb = gen.generate_embedding(face)

    # +10 degrees tilt
    rot_pos = apply_rotation(face, 10.0)
    emb_pos = gen.generate_embedding(rot_pos)
    sim_pos = float(np.dot(base_emb, emb_pos))
    assert sim_pos > 0.75, f"+10 deg tilt similarity {sim_pos} fell below 0.75"

    # -10 degrees tilt
    rot_neg = apply_rotation(face, -10.0)
    emb_neg = gen.generate_embedding(rot_neg)
    sim_neg = float(np.dot(base_emb, emb_neg))
    assert sim_neg > 0.75, f"-10 deg tilt similarity {sim_neg} fell below 0.75"


def test_multi_face_separation():
    """A multi-subject frame must maintain accurate bounding boxes for each subject."""
    face1 = create_synthetic_face(seed=1, size=(100, 100))
    face2 = create_synthetic_face(seed=2, size=(100, 100))
    face3 = create_synthetic_face(seed=3, size=(100, 100))

    positions = [(50, 50), (250, 50), (450, 50)]
    frame, bboxes = create_multi_face_frame([face1, face2, face3], positions)

    assert len(bboxes) == 3
    # Check bounding box coordinates match inputs
    assert bboxes[0] == (50, 50, 100, 100)
    assert bboxes[1] == (250, 50, 100, 100)
    assert bboxes[2] == (450, 50, 100, 100)

    # Crop each face and check dimensions
    for (x, y, w, h) in bboxes:
        crop = frame[y : y + h, x : x + w]
        assert crop.shape == (100, 100, 3)


def test_unknown_impostor_rejection():
    """An unregistered student must be identified as 'UNKNOWN'."""
    matcher = FaceMatcher(cosine_threshold=0.70)

    # Register Student A with unit embedding along axis 0
    emb_a = np.zeros(512, dtype=np.float32)
    emb_a[0] = 1.0
    matcher.register_student("STUDENT_A", emb_a)

    # Impostor Student B with orthogonal unit embedding along axis 1 (cosine sim = 0.0)
    emb_b = np.zeros(512, dtype=np.float32)
    emb_b[1] = 1.0

    result = matcher.match_embedding(emb_b)

    # Must be Unknown because score 0.0 < 0.70 threshold
    assert result.is_known is False
    assert result.confidence < 0.70


def test_adversarial_noise_discrimination():
    """Pure Gaussian noise should yield low similarity against enrolled students."""
    gen = FaceEmbeddingGenerator()
    matcher = FaceMatcher(cosine_threshold=0.70)

    face = create_synthetic_face(seed=42)
    face_emb = gen.generate_embedding(face)
    matcher.register_student("VALID_STUDENT", face_emb)

    noise = create_gaussian_noise(size=(160, 160))
    noise_emb = gen.generate_embedding(noise)

    sim = float(np.dot(face_emb, noise_emb))
    # Cosine similarity with random noise should be far below 0.50
    assert sim < 0.50, f"Adversarial noise similarity {sim} should be < 0.50"


def test_far_frr_benchmarking_calculation():
    """Verify FAR and FRR calculations under synthetic benchmark scores."""
    matcher = FaceMatcher(cosine_threshold=0.70)
    bench = RobustnessBenchmark(matcher)

    # 10 genuine pairs: 9 are >= 0.70, 1 is < 0.70 (FRR = 1/10 = 0.10)
    genuine = [0.85, 0.90, 0.78, 0.72, 0.91, 0.88, 0.75, 0.65, 0.82, 0.79]
    # 10 impostor pairs: 1 is >= 0.70, 9 are < 0.70 (FAR = 1/10 = 0.10)
    impostors = [0.30, 0.42, 0.55, 0.61, 0.73, 0.48, 0.35, 0.51, 0.64, 0.58]

    metrics = bench.compute_far_frr(genuine, impostors, threshold=0.70)
    assert metrics["FAR"] == 0.1
    assert metrics["FRR"] == 0.1
    assert metrics["threshold"] == 0.70
    assert metrics["genuine_count"] == 10
    assert metrics["impostor_count"] == 10
