"""
Unit and integration tests for Module 5 — Face Matching & Recognition.
"""

import pytest
import numpy as np
import cv2

from module5.face_matcher import (
    FaceMatcher,
    MatchResult,
    compute_cosine_similarity,
    compute_euclidean_distance,
    batch_cosine_similarity,
)


@pytest.fixture
def unit_vector():
    """Generate a reproducible random 512-d unit vector."""
    np.random.seed(42)
    v = np.random.randn(512)
    return (v / np.linalg.norm(v)).astype(np.float32)


@pytest.fixture
def populated_matcher():
    """FaceMatcher pre-populated with 3 distinct registered students."""
    matcher = FaceMatcher(cosine_threshold=0.70)
    np.random.seed(123)

    for sid in ["CSE101", "CSE102", "CSE103"]:
        v = np.random.randn(512)
        v = (v / np.linalg.norm(v)).astype(np.float32)
        matcher.register_student(sid, v)

    return matcher


# ==================== METRIC TESTS ====================

def test_cosine_similarity_identical(unit_vector):
    """Cosine similarity of identical vectors must be 1.0."""
    sim = compute_cosine_similarity(unit_vector, unit_vector)
    assert abs(sim - 1.0) < 1e-6


def test_cosine_similarity_orthogonal():
    """Cosine similarity of orthogonal vectors must be 0.0."""
    v1 = np.zeros(512, dtype=np.float32)
    v2 = np.zeros(512, dtype=np.float32)
    v1[0] = 1.0
    v2[1] = 1.0
    sim = compute_cosine_similarity(v1, v2)
    assert abs(sim - 0.0) < 1e-6


def test_cosine_similarity_opposite(unit_vector):
    """Cosine similarity of opposing vectors must be -1.0."""
    opposite = -unit_vector
    sim = compute_cosine_similarity(unit_vector, opposite)
    assert abs(sim - (-1.0)) < 1e-6


def test_euclidean_distance_identical(unit_vector):
    """Euclidean distance of identical vectors must be 0.0."""
    dist = compute_euclidean_distance(unit_vector, unit_vector)
    assert abs(dist - 0.0) < 1e-6


def test_euclidean_distance_relation(unit_vector):
    """Euclidean distance and cosine similarity must satisfy D = sqrt(2 - 2 * cos)."""
    np.random.seed(55)
    v2 = np.random.randn(512)
    v2 = (v2 / np.linalg.norm(v2)).astype(np.float32)

    cos_sim = compute_cosine_similarity(unit_vector, v2)
    euc_dist = compute_euclidean_distance(unit_vector, v2)
    expected_dist = float(np.sqrt(max(0.0, 2.0 - 2.0 * cos_sim)))

    assert abs(euc_dist - expected_dist) < 1e-5


def test_batch_cosine_similarity():
    """Vectorized batch cosine calculation must match individual dot products."""
    np.random.seed(77)
    query = np.random.randn(512)
    query = (query / np.linalg.norm(query)).astype(np.float32)

    gallery = np.random.randn(5, 512).astype(np.float32)
    for i in range(5):
        gallery[i] /= np.linalg.norm(gallery[i])

    batch_sims = batch_cosine_similarity(query, gallery)
    assert batch_sims.shape == (5,)

    for i in range(5):
        single_sim = compute_cosine_similarity(query, gallery[i])
        assert abs(batch_sims[i] - single_sim) < 1e-5


# ==================== MATCHER LOGIC TESTS ====================

def test_matcher_registration():
    """Verify adding, updating, and removing registered students."""
    matcher = FaceMatcher()
    emb = np.ones(512, dtype=np.float32)
    emb /= np.linalg.norm(emb)

    matcher.register_student("STU_A", emb)
    assert "STU_A" in matcher.registered_embeddings
    assert matcher._gallery_matrix.shape == (1, 512)

    # Unregister
    assert matcher.unregister_student("STU_A") is True
    assert "STU_A" not in matcher.registered_embeddings
    assert matcher._gallery_matrix is None


def test_match_known_student(populated_matcher):
    """A query close to CSE102 must be identified as CSE102 with is_known=True."""
    target_emb = populated_matcher.registered_embeddings["CSE102"]

    # Add small noise (95% target + 5% noise)
    np.random.seed(999)
    noise = np.random.randn(512).astype(np.float32)
    noise /= np.linalg.norm(noise)
    query = 0.95 * target_emb + 0.05 * noise
    query /= np.linalg.norm(query)

    result = populated_matcher.match_embedding(query)
    assert result.is_known is True
    assert result.student_id == "CSE102"
    assert result.confidence >= 0.70
    assert result.distance < 0.77


def test_match_unknown_student(populated_matcher):
    """A completely random query vector must be classified as UNKNOWN."""
    np.random.seed(888)
    stranger = np.random.randn(512).astype(np.float32)
    stranger /= np.linalg.norm(stranger)

    result = populated_matcher.match_embedding(stranger)
    assert result.is_known is False
    assert result.student_id == "UNKNOWN"
    assert result.confidence < 0.70


def test_matcher_empty_gallery():
    """An empty matcher must classify any query as UNKNOWN."""
    empty_matcher = FaceMatcher()
    dummy = np.ones(512, dtype=np.float32)
    dummy /= np.linalg.norm(dummy)

    result = empty_matcher.match_embedding(dummy)
    assert result.is_known is False
    assert result.student_id == "UNKNOWN"
    assert result.confidence == 0.0


def test_matcher_threshold_boundary():
    """Verify exact boundary enforcement: >= threshold is accepted, < threshold is rejected."""
    matcher = FaceMatcher(cosine_threshold=0.80)
    base = np.zeros(512, dtype=np.float32)
    base[0] = 1.0
    matcher.register_student("TEST_ID", base)

    # 1. Exact match (sim = 1.0 >= 0.80)
    res_high = matcher.match_embedding(base)
    assert res_high.is_known is True
    assert res_high.student_id == "TEST_ID"

    # 2. Vector with cosine similarity approx 0.70 < 0.80
    v_low = np.zeros(512, dtype=np.float32)
    v_low[0] = 0.70
    v_low[1] = np.sqrt(1.0 - 0.70**2)
    res_low = matcher.match_embedding(v_low)
    assert res_low.is_known is False
    assert res_low.student_id == "UNKNOWN"


def test_draw_matches_overlay():
    """Verify visual annotation renders rectangles and labels without altering image shape."""
    frame = np.full((480, 640, 3), 100, dtype=np.uint8)

    matches = [
        MatchResult(student_id="CSE101", confidence=0.92, distance=0.4, is_known=True, bbox=(50, 50, 100, 100)),
        MatchResult(student_id="UNKNOWN", confidence=0.35, distance=1.1, is_known=False, bbox=(250, 150, 80, 80)),
    ]

    annotated = FaceMatcher.draw_matches(frame, matches)
    assert annotated.shape == frame.shape
    assert not np.array_equal(annotated, frame)  # Frame was modified with annotations
