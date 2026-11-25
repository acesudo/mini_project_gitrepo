"""
Module 5 — Face Matching & Recognition

This module compares query facial embeddings against registered student
representations using Cosine Similarity and Euclidean Distance. It enforces
matching thresholds to accept known students, reject unknown individuals,
and support multiple face recognitions in a single frame.
"""

from .face_matcher import (
    FaceMatcher,
    MatchResult,
    compute_cosine_similarity,
    compute_euclidean_distance,
    batch_cosine_similarity,
)

__all__ = [
    "FaceMatcher",
    "MatchResult",
    "compute_cosine_similarity",
    "compute_euclidean_distance",
    "batch_cosine_similarity",
]
