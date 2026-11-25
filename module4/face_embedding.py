"""
Module 4 — Face Recognition (FaceNet Embeddings)

This module provides the FaceEmbeddingGenerator class, which transforms
preprocessed facial crops (160x160 RGB) into 512-dimensional L2-normalized
embeddings using a pretrained FaceNet (Inception-ResNet-v1) architecture.

It also provides multi-sample representation aggregation for stable student
profiles, batch inference, and serialization utilities for database storage.
"""

from typing import List, Optional, Union
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1


class FaceEmbeddingGenerator:
    """
    Generates 512-dimensional facial embeddings using FaceNet (InceptionResnetV1).
    """

    def __init__(
        self,
        model_name: str = "vggface2",
        device: Optional[Union[str, torch.device]] = None
    ) -> None:
        """
        Initialize the FaceNet embedding generator.

        Args:
            model_name: Pretrained weights to load ('vggface2' or 'casia-webface').
                        Defaults to 'vggface2'.
            device: Computing device ('cuda', 'cpu', or torch.device).
                    If None, automatically detects CUDA availability.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # Load pretrained FaceNet model in evaluation mode
        self.model = InceptionResnetV1(pretrained=model_name).eval()
        self.model.to(self.device)

    def preprocess_image(self, face_image: np.ndarray) -> torch.Tensor:
        """
        Validate and standardize a face image for FaceNet inference.

        Expects a 160x160 RGB image (as produced by Module 3 preprocessing).
        Applies fixed image standardization: (pixel - 127.5) / 128.0.

        Args:
            face_image: RGB image numpy array of shape (160, 160, 3).

        Returns:
            torch.Tensor of shape (1, 3, 160, 160) on self.device.
        """
        if face_image is None or not isinstance(face_image, np.ndarray):
            raise ValueError("Input face_image must be a non-empty numpy.ndarray.")

        if face_image.size == 0:
            raise ValueError("Input face_image is empty.")

        if face_image.shape != (160, 160, 3):
            raise ValueError(
                f"Expected face_image of shape (160, 160, 3), got {face_image.shape}. "
                "Ensure image is preprocessed using module3.face_preprocessing.preprocess_face()."
            )

        # Fixed image standardization: scales [0, 255] to approx [-1, 1]
        standardized = (face_image.astype(np.float32) - 127.5) / 128.0

        # Transpose from (H, W, C) to (C, H, W)
        tensor = torch.from_numpy(standardized).permute(2, 0, 1).float()

        # Add batch dimension: (1, C, H, W) and transfer to target device
        return tensor.unsqueeze(0).to(self.device)

    def generate_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Generate a 512-dimensional L2-normalized embedding for a single face image.

        Args:
            face_image: Preprocessed RGB face image of shape (160, 160, 3).

        Returns:
            np.ndarray: 1D float32 array of shape (512,) with unit L2 norm (||v||_2 = 1.0).
        """
        tensor = self.preprocess_image(face_image)

        with torch.no_grad():
            embedding_tensor = self.model(tensor)

        embedding = embedding_tensor.cpu().numpy().flatten().astype(np.float32)

        # Enforce exact L2 unit normalization
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def generate_embeddings_batch(self, face_images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Generate embeddings for a list of face images.

        Args:
            face_images: List of preprocessed RGB face images.

        Returns:
            List of 512-dimensional float32 embeddings.
        """
        if not face_images:
            return []

        # Validate and prepare batch tensor
        tensors = [self.preprocess_image(img).squeeze(0) for img in face_images]
        batch_tensor = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            batch_embeddings = self.model(batch_tensor).cpu().numpy()

        results = []
        for emb in batch_embeddings:
            flat = emb.astype(np.float32)
            norm = np.linalg.norm(flat)
            if norm > 0:
                flat = flat / norm
            results.append(flat)

        return results

    def generate_student_representation(self, face_images: List[np.ndarray]) -> np.ndarray:
        """
        Create a stable, noise-resistant student face representation from multiple samples.

        Averages the embeddings of all valid face samples (e.g. the 5 samples captured
        during student registration) and projects the centroid back onto the unit hypersphere.

        Args:
            face_images: List of preprocessed RGB face images (min 1 sample).

        Returns:
            np.ndarray: 512-dimensional L2-normalized float32 representation vector.
        """
        if not face_images or len(face_images) == 0:
            raise ValueError("At least one face sample is required to generate student representation.")

        embeddings = [self.generate_embedding(img) for img in face_images]

        # Calculate centroid (mean vector across samples)
        mean_embedding = np.mean(embeddings, axis=0)

        # Re-normalize to unit length
        norm = np.linalg.norm(mean_embedding)
        if norm > 0:
            mean_embedding = mean_embedding / norm

        return mean_embedding.astype(np.float32)

    @staticmethod
    def embedding_to_bytes(embedding: np.ndarray) -> bytes:
        """Convert a 512-d float32 embedding to raw bytes for binary database storage."""
        if not isinstance(embedding, np.ndarray) or embedding.shape != (512,):
            raise ValueError("Expected 512-dimensional numpy array.")
        return embedding.astype(np.float32).tobytes()

    @staticmethod
    def bytes_to_embedding(data: bytes) -> np.ndarray:
        """Reconstruct a 512-d float32 embedding from raw bytes."""
        embedding = np.frombuffer(data, dtype=np.float32)
        if embedding.shape != (512,):
            raise ValueError(f"Expected 512 elements from bytes, got {embedding.shape}.")
        return embedding

    @staticmethod
    def embedding_to_list(embedding: np.ndarray) -> List[float]:
        """Convert embedding to a list of floats for JSON serialization."""
        return embedding.tolist()

    @staticmethod
    def list_to_embedding(data: List[float]) -> np.ndarray:
        """Reconstruct embedding from a list of floats."""
        arr = np.array(data, dtype=np.float32)
        if arr.shape != (512,):
            raise ValueError(f"Expected 512 float values, got {arr.shape}.")
        return arr


def main():
    """Demonstrate Module 4 functionality with synthetic face crops."""
    print("=" * 60)
    print("MODULE 4 — FaceNet Embedding Generation Demo")
    print("=" * 60)

    print("Initializing FaceEmbeddingGenerator...")
    generator = FaceEmbeddingGenerator()
    print(f"Loaded FaceNet model onto device: {generator.device}")

    # Generate sample synthetic face image (160x160x3 uint8)
    np.random.seed(42)
    sample_face_1 = np.random.randint(60, 200, size=(160, 160, 3), dtype=np.uint8)
    sample_face_2 = np.random.randint(80, 220, size=(160, 160, 3), dtype=np.uint8)

    # 1. Single Embedding Generation
    embedding_1 = generator.generate_embedding(sample_face_1)
    norm_1 = np.linalg.norm(embedding_1)
    print("\n--- 1. Single Face Embedding ---")
    print(f"Embedding Shape: {embedding_1.shape}")
    print(f"Embedding Dtype: {embedding_1.dtype}")
    print(f"L2 Norm (should be 1.0): {norm_1:.6f}")
    print(f"First 5 values: {embedding_1[:5]}")

    # 2. Multi-sample Student Representation
    samples = [sample_face_1, sample_face_1.copy(), sample_face_2]
    student_rep = generator.generate_student_representation(samples)
    print("\n--- 2. Multi-Sample Student Representation ---")
    print(f"Student Representation Shape: {student_rep.shape}")
    print(f"Representation L2 Norm: {np.linalg.norm(student_rep):.6f}")

    # 3. Serialization Demonstration
    binary_data = generator.embedding_to_bytes(student_rep)
    restored_embedding = generator.bytes_to_embedding(binary_data)
    print("\n--- 3. Serialization Test ---")
    print(f"Serialized Byte Length: {len(binary_data)} bytes (512 * 4 bytes)")
    print(f"Restoration Match: {np.allclose(student_rep, restored_embedding)}")

    print("\nModule 4 verification completed successfully.")


if __name__ == "__main__":
    main()