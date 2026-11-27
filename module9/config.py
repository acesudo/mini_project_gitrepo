"""
Module 9 — Security Configuration.
Reads settings from environment variables with safe defaults for development.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SecurityConfig:
    SECRET_KEY: str = os.getenv(
        "SECRET_KEY", "face_attendance_default_dev_secret_key_minimum_32_bytes_long_12345"
    )
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
    AUTH_ENABLED: bool = os.getenv("AUTH_ENABLED", "false").lower() in ("true", "1", "yes")
    BIOMETRIC_HMAC_KEY: str = os.getenv(
        "BIOMETRIC_HMAC_KEY", "face_attendance_biometric_hmac_secret_key_default_32"
    )
    INITIAL_ADMIN_USERNAME: str = os.getenv("INITIAL_ADMIN_USERNAME", "admin")
    INITIAL_ADMIN_PASSWORD: str = os.getenv("INITIAL_ADMIN_PASSWORD", "AdminSecurePassword123!")


config = SecurityConfig()
