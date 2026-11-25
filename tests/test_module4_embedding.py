"""
Unit and integration tests for Module 4 — Face Recognition (FaceNet Embeddings).
"""

import pytest
import numpy as np
import torch
from module4.face_embedding import FaceEmbeddingGenerator


@pytest.fixture(scope="module")
def generator():
    """Fixture providing a cached FaceEmbeddingGenerator instance."""
    return FaceEmbeddingGenerator()


def test_generator_initialization(generator):
    """Verify generator initializes with an InceptionResnetV1 model in eval mode."""
    assert generator.model is not None
    assert not generator.model.training  # Model must be in eval() mode
    assert isinstance(generator.device, torch.device)


def test_preprocess_valid_image(generator):
    """Verify preprocessing converts 160x160x3 uint8 array to standardized (1, 3, 160, 160) tensor."""
    dummy_img = np.full((160, 160, 3), 128, dtype=np.uint8)
    tensor = generator.preprocess_image(dummy_img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 160, 160)
    assert tensor.dtype == torch.float32
    assert abs(tensor.mean().item()) < 0.05


def test_preprocess_invalid_image_raises_error(generator):
    """Verify preprocessing rejects None, empty, or incorrectly dimensioned arrays."""
    with pytest.raises(ValueError, match="Input face_image must be a non-empty"):
        generator.preprocess_image(None)

    with pytest.raises(ValueError, match="Input face_image is empty"):
        generator.preprocess_image(np.array([]))

    with pytest.raises(ValueError, match="Expected face_image of shape"):
        generator.preprocess_image(np.zeros((100, 100, 3), dtype=np.uint8))


def test_single_embedding_shape_and_norm(generator):
    """Verify output is 1D float32 of shape (512,) with unit L2 norm."""
    np.random.seed(10)
    face_crop = np.random.randint(50, 200, size=(160, 160, 3), dtype=np.uint8)
    embedding = generator.generate_embedding(face_crop)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (512,)
    assert embedding.dtype == np.float32

    l2_norm = np.linalg.norm(embedding)
    assert abs(l2_norm - 1.0) < 1e-5


def test_embedding_determinism(generator):
    """Verify identical face input produces identical embeddings."""
    np.random.seed(99)
    face_crop = np.random.randint(50, 200, size=(160, 160, 3), dtype=np.uint8)

    emb1 = generator.generate_embedding(face_crop)
    emb2 = generator.generate_embedding(face_crop)

    assert np.allclose(emb1, emb2, atol=1e-6)


def test_embedding_discrimination(generator):
    """Verify distinct visual patterns yield distinguishable embeddings."""
    pattern_dark = np.full((160, 160, 3), 60, dtype=np.uint8)
    pattern_bright = np.full((160, 160, 3), 210, dtype=np.uint8)

    emb_dark = generator.generate_embedding(pattern_dark)
    emb_bright = generator.generate_embedding(pattern_bright)

    cosine_sim = np.dot(emb_dark, emb_bright)
    assert cosine_sim < 0.99


def test_batch_embeddings(generator):
    """Verify batch embedding generation produces expected outputs."""
    np.random.seed(123)
    imgs = [
        np.random.randint(50, 200, size=(160, 160, 3), dtype=np.uint8)
        for _ in range(3)
    ]
    batch_embeddings = generator.generate_embeddings_batch(imgs)

    assert len(batch_embeddings) == 3
    for emb in batch_embeddings:
        assert emb.shape == (512,)
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-5


def test_student_representation_aggregation(generator):
    """Verify averaging multiple samples creates a normalized 512-d vector."""
    np.random.seed(42)
    base = np.random.randint(80, 180, size=(160, 160, 3), dtype=np.uint8)
    samples = [
        np.clip(base + np.random.randint(-10, 10, size=base.shape), 0, 255).astype(np.uint8)
        for _ in range(5)
    ]

    student_rep = generator.generate_student_representation(samples)

    assert isinstance(student_rep, np.ndarray)
    assert student_rep.shape == (512,)
    assert student_rep.dtype == np.float32
    assert abs(np.linalg.norm(student_rep) - 1.0) < 1e-5


def test_student_representation_empty_raises_error(generator):
    """Verify empty sample list raises ValueError."""
    with pytest.raises(ValueError, match="At least one face sample is required"):
        generator.generate_student_representation([])


def test_serialization_binary(generator):
    """Verify binary byte serialization and deserialization."""
    np.random.seed(7)
    face = np.random.randint(60, 200, size=(160, 160, 3), dtype=np.uint8)
    original = generator.generate_embedding(face)

    data_bytes = generator.embedding_to_bytes(original)
    assert isinstance(data_bytes, bytes)
    assert len(data_bytes) == 512 * 4  # 512 float32 values = 2048 bytes

    restored = generator.bytes_to_embedding(data_bytes)
    assert np.array_equal(original, restored)


def test_serialization_list(generator):
    """Verify list/JSON serialization and deserialization."""
    np.random.seed(8)
    face = np.random.randint(60, 200, size=(160, 160, 3), dtype=np.uint8)
    original = generator.generate_embedding(face)

    data_list = generator.embedding_to_list(original)
    assert isinstance(data_list, list)
    assert len(data_list) == 512

    restored = generator.list_to_embedding(data_list)
    assert np.allclose(original, restored)
