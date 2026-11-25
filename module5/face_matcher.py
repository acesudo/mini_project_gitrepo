"""
Module 5 — Face Matching Engine

Provides the FaceMatcher class for comparing query embeddings against a registered
student gallery using Cosine Similarity and Euclidean Distance. Supports threshold
evaluation, unknown face handling, and multi-face frame recognition.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import cv2

# Ensure project root is in sys.path for direct script execution
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from module2.harr_classifier import detect_faces
from module3.face_preprocessing import preprocess_face
from module4.face_embedding import FaceEmbeddingGenerator


@dataclass
class MatchResult:
    """Represents the outcome of a facial embedding comparison."""
    student_id: Optional[str]
    confidence: float
    distance: float
    is_known: bool
    bbox: Optional[Tuple[int, int, int, int]] = None

    def to_dict(self) -> dict:
        return {
            "student_id": self.student_id,
            "confidence": round(float(self.confidence), 4),
            "distance": round(float(self.distance), 4),
            "is_known": self.is_known,
            "bbox": self.bbox,
        }


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate Cosine Similarity between two 512-dimensional embeddings.

    For L2-normalized vectors (||v||_2 = 1.0), this equals their dot product:
        cosine_similarity = dot(vec1, vec2)
    Range: [-1.0, 1.0], where 1.0 indicates identical orientation.
    """
    if vec1 is None or vec2 is None:
        raise ValueError("Embeddings cannot be None.")
    if vec1.shape != (512,) or vec2.shape != (512,):
        raise ValueError(f"Expected shape (512,), got {vec1.shape} and {vec2.shape}.")

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    # Ensure vectors are normalized
    v1 = vec1 / norm1
    v2 = vec2 / norm2

    dot_product = float(np.dot(v1, v2))
    return float(np.clip(dot_product, -1.0, 1.0))


def compute_euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate Euclidean (L2) distance between two 512-dimensional embeddings.

    For unit-normalized vectors:
        distance = sqrt(2 - 2 * cosine_similarity)
    Range: [0.0, 2.0], where 0.0 indicates identical vectors.
    """
    if vec1 is None or vec2 is None:
        raise ValueError("Embeddings cannot be None.")
    if vec1.shape != (512,) or vec2.shape != (512,):
        raise ValueError(f"Expected shape (512,), got {vec1.shape} and {vec2.shape}.")

    diff = vec1.astype(np.float32) - vec2.astype(np.float32)
    return float(np.linalg.norm(diff))


def batch_cosine_similarity(query: np.ndarray, gallery_matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarities between one query vector and N gallery vectors in a single operation.

    Args:
        query: 1D float32 array of shape (512,).
        gallery_matrix: 2D float32 array of shape (N, 512).

    Returns:
        1D float32 array of shape (N,) containing similarity scores.
    """
    if query.ndim != 1 or query.shape[0] != 512:
        raise ValueError(f"Expected query of shape (512,), got {query.shape}.")
    if gallery_matrix.ndim != 2 or gallery_matrix.shape[1] != 512:
        raise ValueError(f"Expected gallery of shape (N, 512), got {gallery_matrix.shape}.")

    # Matrix-vector product E * q
    scores = np.dot(gallery_matrix, query)
    return np.clip(scores, -1.0, 1.0)


class FaceMatcher:
    """
    Facial matching engine comparing query embeddings against a registered gallery.
    """

    def __init__(
        self,
        registered_embeddings: Optional[Dict[str, np.ndarray]] = None,
        cosine_threshold: float = 0.70,
        generator: Optional[FaceEmbeddingGenerator] = None,
    ) -> None:
        """
        Initialize the FaceMatcher.

        Args:
            registered_embeddings: Optional dictionary mapping student_id to 512-d unit embedding.
            cosine_threshold: Minimum cosine similarity required to accept a match (default: 0.70).
            generator: Optional pre-instantiated FaceEmbeddingGenerator.
        """
        if not 0.0 <= cosine_threshold <= 1.0:
            raise ValueError("cosine_threshold must be between 0.0 and 1.0.")

        self.cosine_threshold = cosine_threshold
        self._generator = generator

        self.registered_embeddings: Dict[str, np.ndarray] = {}
        self._gallery_ids: List[str] = []
        self._gallery_matrix: Optional[np.ndarray] = None

        if registered_embeddings:
            self.register_students(registered_embeddings)

    @property
    def generator(self) -> FaceEmbeddingGenerator:
        """Lazy-load FaceEmbeddingGenerator if not provided during init."""
        if self._generator is None:
            self._generator = FaceEmbeddingGenerator()
        return self._generator

    def register_student(self, student_id: str, embedding: np.ndarray) -> None:
        """
        Add or update a student's registered face representation.

        Args:
            student_id: Unique student identifier.
            embedding: 512-dimensional float32 vector.
        """
        student_id = str(student_id).strip().upper()
        if not student_id:
            raise ValueError("student_id cannot be empty.")

        if not isinstance(embedding, np.ndarray) or embedding.shape != (512,):
            raise ValueError(f"Expected 512-dimensional embedding, got {embedding.shape}.")

        # Enforce exact unit norm
        norm = np.linalg.norm(embedding)
        if norm == 0:
            raise ValueError("Cannot register an all-zero embedding.")
        unit_embedding = (embedding / norm).astype(np.float32)

        self.registered_embeddings[student_id] = unit_embedding
        self._update_gallery_cache()

    def register_students(self, students: Dict[str, np.ndarray]) -> None:
        """Register multiple students in bulk."""
        for sid, emb in students.items():
            norm = np.linalg.norm(emb)
            if norm > 0:
                self.registered_embeddings[str(sid).strip().upper()] = (emb / norm).astype(np.float32)
        self._update_gallery_cache()

    def unregister_student(self, student_id: str) -> bool:
        """Remove a student from the registered gallery."""
        sid = str(student_id).strip().upper()
        if sid in self.registered_embeddings:
            del self.registered_embeddings[sid]
            self._update_gallery_cache()
            return True
        return False

    def _update_gallery_cache(self) -> None:
        """Rebuild internal gallery matrix for fast vectorized operations."""
        if not self.registered_embeddings:
            self._gallery_ids = []
            self._gallery_matrix = None
            return

        self._gallery_ids = list(self.registered_embeddings.keys())
        matrix = np.array([self.registered_embeddings[sid] for sid in self._gallery_ids], dtype=np.float32)
        self._gallery_matrix = matrix

    def match_embedding(self, query_embedding: np.ndarray) -> MatchResult:
        """
        Match a query embedding against the gallery of registered students.

        Args:
            query_embedding: 512-dimensional float32 vector.

        Returns:
            MatchResult containing student_id, confidence, distance, and is_known flag.
        """
        if not isinstance(query_embedding, np.ndarray) or query_embedding.shape != (512,):
            raise ValueError(f"Expected query_embedding of shape (512,), got {query_embedding.shape}.")

        # If no students registered, all queries are Unknown
        if not self.registered_embeddings or self._gallery_matrix is None:
            return MatchResult(
                student_id="UNKNOWN",
                confidence=0.0,
                distance=2.0,
                is_known=False,
            )

        # Ensure query is unit normalized
        q_norm = np.linalg.norm(query_embedding)
        if q_norm == 0:
            return MatchResult(
                student_id="UNKNOWN",
                confidence=0.0,
                distance=2.0,
                is_known=False,
            )
        q = (query_embedding / q_norm).astype(np.float32)

        # Vectorized similarity calculation E * q
        similarities = batch_cosine_similarity(q, self._gallery_matrix)

        best_index = int(np.argmax(similarities))
        best_confidence = float(similarities[best_index])
        best_id = self._gallery_ids[best_index]

        # Convert cosine similarity to Euclidean distance: sqrt(2 - 2 * sim)
        euclidean_dist = float(np.sqrt(max(0.0, 2.0 - 2.0 * best_confidence)))

        if best_confidence >= self.cosine_threshold:
            return MatchResult(
                student_id=best_id,
                confidence=best_confidence,
                distance=euclidean_dist,
                is_known=True,
            )
        else:
            return MatchResult(
                student_id="UNKNOWN",
                confidence=best_confidence,
                distance=euclidean_dist,
                is_known=False,
            )

    def match_frame(self, frame: np.ndarray) -> List[MatchResult]:
        """
        Process a complete camera frame: detect faces, extract embeddings, and match all candidates.

        Args:
            frame: Raw BGR camera frame from OpenCV.

        Returns:
            List of MatchResult objects with bounding box coordinates.
        """
        if frame is None or frame.size == 0:
            return []

        # Step 1: Detect all faces in frame using Module 2 Haar Cascade
        faces = detect_faces(frame)
        if len(faces) == 0:
            return []

        results: List[MatchResult] = []

        # Step 2: Process each detected face independently
        for bbox in faces:
            x, y, w, h = map(int, bbox)

            try:
                # Step 3: Crop & resize to 160x160 RGB using Module 3
                face_crop = preprocess_face(frame, bbox, size=(160, 160), padding=0.15)

                # Step 4: Generate 512-d embedding using Module 4
                query_emb = self.generator.generate_embedding(face_crop)

                # Step 5: Match against gallery using Module 5
                match = self.match_embedding(query_emb)
                match.bbox = (x, y, w, h)
                results.append(match)

            except Exception as e:
                # Malformed or edge crop: mark as unknown with zero confidence
                results.append(
                    MatchResult(
                        student_id="UNKNOWN",
                        confidence=0.0,
                        distance=2.0,
                        is_known=False,
                        bbox=(x, y, w, h),
                    )
                )

        return results

    @staticmethod
    def draw_matches(frame: np.ndarray, matches: List[MatchResult]) -> np.ndarray:
        """
        Annotate a frame with color-coded bounding boxes and student recognition tags.

        - Known Student: Green bounding box with Student ID and confidence score.
        - Unknown Person: Red bounding box with 'Unknown' tag and confidence score.
        """
        output = frame.copy()

        for match in matches:
            if match.bbox is None:
                continue

            x, y, w, h = match.bbox

            if match.is_known:
                color = (0, 255, 0)  # Green for accepted match
                label = f"{match.student_id} ({match.confidence:.2f})"
            else:
                color = (0, 0, 255)  # Red for rejected / unknown
                label = f"Unknown ({match.confidence:.2f})"

            # Draw face bounding rectangle
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

            # Draw background banner for label text readability
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(
                output,
                (x, max(0, y - text_h - 10)),
                (x + text_w + 6, max(0, y)),
                color,
                cv2.FILLED,
            )

            # Draw white label text
            cv2.putText(
                output,
                label,
                (x + 3, max(0, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

        return output


def main():
    """Demonstrate Module 5 matching functionality with synthetic student embeddings."""
    print("=" * 60)
    print("MODULE 5 — Face Matching Engine Demo")
    print("=" * 60)

    # 1. Initialize Matcher with Cosine Threshold of 0.70
    matcher = FaceMatcher(cosine_threshold=0.70)

    # 2. Register mock students with unique random unit vectors
    np.random.seed(101)
    student_a_emb = np.random.randn(512)
    student_a_emb /= np.linalg.norm(student_a_emb)

    student_b_emb = np.random.randn(512)
    student_b_emb /= np.linalg.norm(student_b_emb)

    matcher.register_student("CSE1025", student_a_emb)
    matcher.register_student("CSE1026", student_b_emb)
    print(f"Registered {len(matcher.registered_embeddings)} students: {list(matcher.registered_embeddings.keys())}")

    # 3. Test Match 1: Query is a slightly perturbed sample of Student A (Known match)
    noise = np.random.randn(512)
    noise /= np.linalg.norm(noise)
    query_a = 0.90 * student_a_emb + 0.10 * noise
    query_a /= np.linalg.norm(query_a)
    result_a = matcher.match_embedding(query_a)

    print("\n--- Test 1: Query of Student A (Slight noise) ---")
    print(f"Matched Student ID: {result_a.student_id}")
    print(f"Cosine Similarity (Confidence): {result_a.confidence:.4f}")
    print(f"Euclidean Distance: {result_a.distance:.4f}")
    print(f"Is Recognized Known Student: {result_a.is_known}")

    # 4. Test Match 2: Query is an unregistered stranger (Unknown match)
    stranger_emb = np.random.randn(512)
    stranger_emb /= np.linalg.norm(stranger_emb)
    result_stranger = matcher.match_embedding(stranger_emb)

    print("\n--- Test 2: Unregistered Stranger ---")
    print(f"Matched Student ID: {result_stranger.student_id}")
    print(f"Best Gallery Similarity: {result_stranger.confidence:.4f}")
    print(f"Is Recognized Known Student: {result_stranger.is_known}")

    print("\nModule 5 demonstration completed successfully.")


if __name__ == "__main__":
    main()
