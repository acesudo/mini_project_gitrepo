"""
Module 9 — Security, Authentication & Privacy Tests.

Tests:
1. Password hashing & verification with bcrypt.
2. JWT token creation, signature validation, expiration, and invalid token rejection.
3. Biometric HMAC signing and tampering detection on face embeddings.
4. SecurityHeadersMiddleware (X-Frame-Options, X-Content-Type-Options, etc.).
5. Auth API endpoints:
   - Login success & token issuance
   - Login failure (bad password, non-existent user)
   - /auth/me returns authenticated user info
   - User creation by admin succeeds
   - User creation by non-admin is forbidden (403)
   - Listing users by admin
   - Duplicate username rejection (409)
"""

from datetime import timedelta
import pytest
import numpy as np
import jwt
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from module6.attendance_manager import AttendanceManager
from module5.face_matcher import FaceMatcher
from module7.app import create_app
from module9.security import hash_password, verify_password, create_access_token, decode_access_token
from module9.biometric_crypto import BiometricSigner
from module9.config import config


@pytest.fixture
def auth_client(tmp_path):
    """Test client with isolated SQLite database and seeded admin."""
    test_db = tmp_path / "test_sec.db"
    engine = create_engine(f"sqlite:///{test_db}")
    manager = AttendanceManager(engine=engine)
    matcher = FaceMatcher()
    app = create_app(engine=engine, manager=manager, matcher=matcher)
    return TestClient(app)


# --- 1. Password Security Tests ---

def test_password_hashing_and_verification():
    plain = "SuperSecurePassword123!"
    hashed = hash_password(plain)

    # Hash should be non-empty and different from plain
    assert hashed != plain
    assert hashed.startswith("$2b$") or hashed.startswith("$2a$")

    # Correct password verifies
    assert verify_password(plain, hashed) is True

    # Incorrect password fails
    assert verify_password("WrongPassword!", hashed) is False
    assert verify_password("", hashed) is False


def test_password_hashing_empty_raises_error():
    with pytest.raises(ValueError):
        hash_password("")


# --- 2. JWT Token Tests ---

def test_jwt_token_lifecycle():
    data = {"sub": "test_teacher", "role": "teacher", "uid": 42}
    token = create_access_token(data, expires_delta=timedelta(minutes=15))

    # Token can be decoded
    payload = decode_access_token(token)
    assert payload["sub"] == "test_teacher"
    assert payload["role"] == "teacher"
    assert payload["uid"] == 42
    assert "exp" in payload
    assert "iat" in payload


def test_jwt_token_expired():
    data = {"sub": "expired_user"}
    # Token expired 5 minutes ago
    token = create_access_token(data, expires_delta=timedelta(minutes=-5))

    with pytest.raises(jwt.ExpiredSignatureError):
        decode_access_token(token)


def test_jwt_token_invalid_signature():
    token = create_access_token({"sub": "user1"}, secret_key="correct_secret_key_1234567890123456")

    with pytest.raises(jwt.InvalidSignatureError):
        decode_access_token(token, secret_key="wrong_secret_key_123456789012345678")


# --- 3. Biometric HMAC Integrity Tests ---

def test_biometric_signer_integrity():
    signer = BiometricSigner(hmac_key="secret_biometric_key_12345")
    # Synthetic 512-d embedding
    embedding = np.random.randn(512).astype(np.float32)
    student_id = "STU_999"

    # Compute signature
    signature = signer.sign_embedding(embedding, student_id)
    assert isinstance(signature, str)
    assert len(signature) == 64  # SHA256 hex digest is 64 chars

    # Verifies cleanly with untampered vector
    assert signer.verify_embedding(embedding, student_id, signature) is True

    # Fails if student_id is swapped
    assert signer.verify_embedding(embedding, "STU_DIFFERENT", signature) is False

    # Fails if even one floating-point value is tampered
    tampered_embedding = embedding.copy()
    tampered_embedding[0] += 0.001
    assert signer.verify_embedding(tampered_embedding, student_id, signature) is False


# --- 4. Security Headers Middleware Tests ---

def test_security_headers_present(auth_client):
    response = auth_client.get("/health")
    assert response.status_code == 200

    headers = response.headers
    assert headers.get("X-Content-Type-Options") == "nosniff"
    assert headers.get("X-Frame-Options") == "DENY"
    assert headers.get("X-XSS-Protection") == "1; mode=block"
    assert headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
    assert "Strict-Transport-Security" in headers


# --- 5. Auth API Endpoints Tests ---

def test_login_success(auth_client):
    response = auth_client.post(
        "/api/v1/auth/login",
        json={"username": config.INITIAL_ADMIN_USERNAME, "password": config.INITIAL_ADMIN_PASSWORD},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["role"] == "admin"
    assert data["username"] == config.INITIAL_ADMIN_USERNAME


def test_login_invalid_credentials(auth_client):
    response = auth_client.post(
        "/api/v1/auth/login",
        json={"username": "admin", "password": "WrongPasswordXYZ!"},
    )
    assert response.status_code == 401
    assert "Incorrect username or password" in response.json()["detail"]


def test_auth_me_endpoint(auth_client):
    # 1. Login to get token
    login_resp = auth_client.post(
        "/api/v1/auth/login",
        json={"username": config.INITIAL_ADMIN_USERNAME, "password": config.INITIAL_ADMIN_PASSWORD},
    )
    token = login_resp.json()["access_token"]

    # 2. Call /me with Bearer token
    me_resp = auth_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert me_resp.status_code == 200
    me_data = me_resp.json()
    assert me_data["username"] == config.INITIAL_ADMIN_USERNAME
    assert me_data["role"] == "admin"


def test_admin_creates_teacher_and_rbac(auth_client):
    # 1. Login as admin
    login_resp = auth_client.post(
        "/api/v1/auth/login",
        json={"username": config.INITIAL_ADMIN_USERNAME, "password": config.INITIAL_ADMIN_PASSWORD},
    )
    admin_token = login_resp.json()["access_token"]

    # 2. Admin creates a new teacher
    create_resp = auth_client.post(
        "/api/v1/auth/users",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={
            "username": "prof_sharma",
            "password": "TeacherPassword123!",
            "role": "teacher",
            "full_name": "Prof. Sharma",
        },
    )
    assert create_resp.status_code == 201
    teacher_data = create_resp.json()
    assert teacher_data["username"] == "prof_sharma"
    assert teacher_data["role"] == "teacher"

    # 3. Duplicate username fails with 409
    dup_resp = auth_client.post(
        "/api/v1/auth/users",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={
            "username": "prof_sharma",
            "password": "AnotherPassword123!",
            "role": "teacher",
        },
    )
    assert dup_resp.status_code == 409

    # 4. Teacher logs in
    teacher_login = auth_client.post(
        "/api/v1/auth/login",
        json={"username": "prof_sharma", "password": "TeacherPassword123!"},
    )
    assert teacher_login.status_code == 200
    teacher_token = teacher_login.json()["access_token"]

    # 5. Teacher attempts to create another user -> MUST FAIL with 403 Forbidden
    forbidden_resp = auth_client.post(
        "/api/v1/auth/users",
        headers={"Authorization": f"Bearer {teacher_token}"},
        json={"username": "kiosk_device_1", "password": "KioskPassword123!", "role": "kiosk"},
    )
    assert forbidden_resp.status_code == 403
    assert "lacks permission" in forbidden_resp.json()["detail"]

    # 6. Admin can list users
    list_resp = auth_client.get(
        "/api/v1/auth/users",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert list_resp.status_code == 200
    users = list_resp.json()
    usernames = [u["username"] for u in users]
    assert config.INITIAL_ADMIN_USERNAME in usernames
    assert "prof_sharma" in usernames
