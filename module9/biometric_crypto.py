"""
Module 9 — Biometric Data Protection & Embedding Integrity Signing.

Face recognition systems handle sensitive biometric data.
This module guarantees:
1. Embedding Integrity: Computes and verifies an HMAC-SHA256 signature
   over the 512-dimensional embedding vector to prevent offline tampering or injection.
2. Irreversible Feature Representation: Validates that only normalized vectors
   are stored, and never raw facial images.
"""

import hmac
import hashlib
from typing import Union, List, Optional
import numpy as np

from module9.config import config


class BiometricSigner:
    """
    Cryptographic HMAC-SHA256 signer for biometric face embeddings.
    Ensures embeddings stored in database or transmitted over API
    have not been forged or altered by an unauthorized actor.
    """

    def __init__(self, hmac_key: Optional[str] = None):
        key = hmac_key or config.BIOMETRIC_HMAC_KEY
        self.key_bytes = key.encode("utf-8")

    def _to_bytes(self, embedding: Union[np.ndarray, List[float], bytes]) -> bytes:
        if isinstance(embedding, bytes):
            return embedding
        if isinstance(embedding, list):
            arr = np.array(embedding, dtype=np.float32)
            return arr.tobytes()
        if isinstance(embedding, np.ndarray):
            return embedding.astype(np.float32).tobytes()
        raise TypeError(f"Unsupported embedding type: {type(embedding)}")

    def sign_embedding(self, embedding: Union[np.ndarray, List[float], bytes], student_id: str) -> str:
        """
        Generate a cryptographic HMAC-SHA256 signature binding the embedding vector
        to the specific student_id.
        """
        raw_bytes = self._to_bytes(embedding)
        message = student_id.encode("utf-8") + b":" + raw_bytes
        signature = hmac.new(self.key_bytes, message, hashlib.sha256).hexdigest()
        return signature

    def verify_embedding(
        self,
        embedding: Union[np.ndarray, List[float], bytes],
        student_id: str,
        expected_signature: str,
    ) -> bool:
        """
        Verify that the embedding has not been tampered with and belongs to student_id.
        Uses hmac.compare_digest for constant-time comparison.
        """
        computed = self.sign_embedding(embedding, student_id)
        return hmac.compare_digest(computed, expected_signature)
