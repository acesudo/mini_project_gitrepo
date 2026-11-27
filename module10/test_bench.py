"""
Module 10 — Robustness Benchmark Runner & Accuracy Metrics Calculator.

Computes:
- Similarity degradation across lighting factors and pose rotations.
- FAR (False Acceptance Rate) and FRR (False Rejection Rate) at threshold tau.
"""

from typing import Dict, List, Tuple
import numpy as np

from module10.synthetic_data import create_synthetic_face, apply_lighting, apply_rotation
from module5.face_matcher import FaceMatcher


class RobustnessBenchmark:
    """
    Runs automated sensitivity benchmarks across lighting, pose, and impostors.
    """

    def __init__(self, matcher: FaceMatcher):
        self.matcher = matcher

    def benchmark_lighting_sensitivity(
        self,
        base_embedding: np.ndarray,
        light_factors: List[float] = [0.5, 0.75, 1.0, 1.25, 1.5],
    ) -> List[Dict[str, float]]:
        """
        Calculates cosine similarity of base embedding against synthetic lighting variations.
        """
        results = []
        for factor in light_factors:
            # Generate lighting-shifted face
            face = create_synthetic_face(seed=42)
            face_lit = apply_lighting(face, factor)
            # Embed face
            emb = self.matcher.embedding_generator.generate_embedding(face_lit)
            sim = float(self.matcher.compute_cosine_similarity(base_embedding, emb))
            results.append({"lighting_factor": factor, "similarity": round(sim, 4)})
        return results

    def benchmark_pose_sensitivity(
        self,
        base_embedding: np.ndarray,
        angles: List[float] = [-30.0, -15.0, 0.0, 15.0, 30.0],
    ) -> List[Dict[str, float]]:
        """
        Calculates cosine similarity across head tilt rotations.
        """
        results = []
        for deg in angles:
            face = create_synthetic_face(seed=42)
            face_rot = apply_rotation(face, deg)
            emb = self.matcher.embedding_generator.generate_embedding(face_rot)
            sim = float(self.matcher.compute_cosine_similarity(base_embedding, emb))
            results.append({"angle_degrees": deg, "similarity": round(sim, 4)})
        return results

    def compute_far_frr(
        self,
        genuine_scores: List[float],
        impostor_scores: List[float],
        threshold: float = 0.70,
    ) -> Dict[str, float]:
        """
        Computes False Acceptance Rate (FAR) and False Rejection Rate (FRR).
        - FAR: Fraction of impostor pairs with score >= threshold
        - FRR: Fraction of genuine pairs with score < threshold
        """
        g_arr = np.array(genuine_scores)
        i_arr = np.array(impostor_scores)

        frr = float(np.mean(g_arr < threshold)) if len(g_arr) > 0 else 0.0
        far = float(np.mean(i_arr >= threshold)) if len(i_arr) > 0 else 0.0

        return {
            "threshold": threshold,
            "FAR": round(far, 4),
            "FRR": round(frr, 4),
            "genuine_count": len(g_arr),
            "impostor_count": len(i_arr),
        }
