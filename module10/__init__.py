"""
Module 10 — Comprehensive Testing Framework & Benchmark Suite.
"""

from module10.synthetic_data import (
    create_synthetic_face,
    apply_lighting,
    apply_rotation,
    create_multi_face_frame,
    create_gaussian_noise,
)
from module10.test_bench import RobustnessBenchmark

__all__ = [
    "create_synthetic_face",
    "apply_lighting",
    "apply_rotation",
    "create_multi_face_frame",
    "create_gaussian_noise",
    "RobustnessBenchmark",
]
