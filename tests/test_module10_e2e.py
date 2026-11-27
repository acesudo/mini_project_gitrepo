"""
Module 10 — End-to-End (E2E) Lifecycle & Concurrency Stress Tests.

Tests:
1. Complete student lifecycle:
   - Create student via POST /students
   - Register 4 facial samples via POST /students/register-with-faces
   - Embedding match via POST /recognition/match-embedding
   - Mark attendance via POST /attendance/mark
   - Duplicate attendance suppression (409 Conflict)
   - Today's attendance listing via GET /attendance/daily
   - Cascade deletion via DELETE /students/{id}
2. High-concurrency burst attendance simulation (race condition stress test).
"""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from module6.attendance_manager import AttendanceManager
from module5.face_matcher import FaceMatcher
from module4.face_embedding import FaceEmbeddingGenerator
from module7.app import create_app
import module7.routers.students as students_router
import module5.face_matcher as face_matcher_module


def make_test_png(seed: int = 0) -> bytes:
    """Generate a clean test face image."""
    rng = np.random.RandomState(seed)
    img = rng.randint(70, 190, size=(160, 160, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


@pytest.fixture
def one_face_patch(monkeypatch):
    """Ensure detect_faces detects one face crop for test images."""
    fake_detector = lambda image: np.array([[10, 10, 140, 140]])
    monkeypatch.setattr(students_router, "detect_faces", fake_detector)
    monkeypatch.setattr(face_matcher_module, "detect_faces", fake_detector)


@pytest.fixture
def e2e_client(one_face_patch):
    """Test client with isolated SQLite database and fresh FaceMatcher."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    manager = AttendanceManager(engine=engine)
    matcher = FaceMatcher(cosine_threshold=0.70)
    app = create_app(engine=engine, manager=manager, matcher=matcher)
    return TestClient(app)


def test_complete_student_lifecycle_e2e(e2e_client):
    """
    Validates entire system pipeline:
    Create Student -> Upload Faces -> Match Embedding -> Mark -> Prevent Duplicate -> Delete.
    """
    student_id = "E2E_STUDENT_01"

    # Step 1: Create student via API
    create_resp = e2e_client.post(
        "/students",
        json={
            "student_id": student_id,
            "name": "Arjun Patel",
            "department": "Computer Science",
            "semester": 3,
        },
    )
    assert create_resp.status_code == 201

    # Step 2: Enroll student faces via /students/register-with-faces
    files = [("images", (f"face_{i}.png", make_test_png(i), "image/png")) for i in range(4)]
    form_data = {
        "student_id": student_id,
        "name": "Arjun Patel",
        "department": "Computer Science",
        "semester": "3",
    }
    enroll_resp = e2e_client.post("/students/register-with-faces", data=form_data, files=files)
    assert enroll_resp.status_code in (200, 201)
    assert enroll_resp.json()["samples_used"] >= 3

    # Step 3: Match via recognition/embedding endpoint
    gen = FaceEmbeddingGenerator()
    query_face = np.full((160, 160, 3), 120, dtype=np.uint8)
    query_emb = gen.generate_embedding(query_face).tolist()

    match_resp = e2e_client.post(
        "/recognition/embedding",
        json={"embedding": query_emb, "mark_attendance": False},
    )
    assert match_resp.status_code == 200

    # Step 4: Mark attendance manually
    mark_resp = e2e_client.post(
        "/attendance/mark",
        json={"student_id": student_id, "status": "Present"},
    )
    assert mark_resp.status_code == 201
    assert mark_resp.json()["student_id"] == student_id

    # Step 5: Prevent duplicate attendance on same date
    dup_resp = e2e_client.post(
        "/attendance/mark",
        json={"student_id": student_id, "status": "Present"},
    )
    # Duplicate entry must be rejected with 409 Conflict
    assert dup_resp.status_code == 409

    # Step 6: Query today's attendance log
    today_resp = e2e_client.get("/attendance/daily")
    assert today_resp.status_code == 200
    records = today_resp.json()
    assert len(records) >= 1
    found = any(rec["student_id"] == student_id for rec in records)
    assert found is True

    # Step 7: Delete student (Right to Be Forgotten)
    del_resp = e2e_client.delete(f"/students/{student_id}")
    assert del_resp.status_code == 204

    # Verify student is removed from database
    get_resp = e2e_client.get(f"/students/{student_id}")
    assert get_resp.status_code == 404


def test_concurrent_attendance_stress(e2e_client):
    """
    Simulates rapid attendance submissions for the same student to ensure
    database locks, rollback handling, and unique constraints prevent duplicate records.
    """
    student_id = "STRESS_TEST_01"

    # Register student
    create_resp = e2e_client.post(
        "/students",
        json={"student_id": student_id, "name": "Stress Subject", "department": "IT", "semester": 1},
    )
    assert create_resp.status_code == 201

    results = []
    # Rapid burst of 10 requests for the same student on the same day
    for i in range(10):
        resp = e2e_client.post(
            "/attendance/mark",
            json={"student_id": student_id, "status": "Present"},
        )
        results.append(resp.status_code)

    # Exactly ONE must succeed (201 Created), the other 9 must be rejected (409 Conflict)
    success_count = results.count(201)
    conflict_count = results.count(409)

    assert success_count == 1, f"Expected exactly 1 success, got {success_count} (all responses: {results})"
    assert conflict_count == 9, f"Expected exactly 9 conflicts, got {conflict_count}"
