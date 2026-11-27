"""
Module 9 — Security Primitives: Password Hashing and JWT Token Management.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
import bcrypt
import jwt

from module9.config import config


def hash_password(password: str) -> str:
    """
    Hash a plaintext password using bcrypt with automatic salt generation.
    Returns the decoded UTF-8 hash string suitable for database storage.
    """
    if not password:
        raise ValueError("Password cannot be empty.")
    pwd_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(pwd_bytes, salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plaintext password against a stored bcrypt hash.
    Safe against timing attacks.
    """
    if not plain_password or not hashed_password:
        return False
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except Exception:
        return False


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
    secret_key: Optional[str] = None,
    algorithm: Optional[str] = None,
) -> str:
    """
    Create a signed JWT access token containing subject and custom claims.
    """
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire, "iat": now})
    key = secret_key or config.SECRET_KEY
    alg = algorithm or config.ALGORITHM
    encoded_jwt = jwt.encode(to_encode, key, algorithm=alg)
    return encoded_jwt


def decode_access_token(
    token: str,
    secret_key: Optional[str] = None,
    algorithm: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Decode and validate a JWT access token.
    Raises jwt.PyJWTError (e.g. ExpiredSignatureError, InvalidSignatureError) if invalid.
    """
    key = secret_key or config.SECRET_KEY
    alg = algorithm or config.ALGORITHM
    payload = jwt.decode(token, key, algorithms=[alg])
    return payload
