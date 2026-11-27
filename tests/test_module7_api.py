"""
Tests for Module 7 — FastAPI Backend.

Uses an isolated in-memory SQLite database and a fake FaceNet generator, so the
tests are fast and need no webcam, no model download and no real faces.
"""

from datetime import date, timedelta

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

import module5.face_matcher as face_matcher_module
import module7.routers.students as students_router
from module5.face_matcher import FaceMatcher
from module6.attendance_manager import AttendanceManager
from module7.app import create_app


# ----------------------------------------------------------------- helpers
def unit_vector(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(512).astype(np.float32)
    return v / np.linalg.norm(v)


class FakeGenerator:
    """Stands in for FaceEmbeddingGenerator. Returns whatever `embedding` is set to."""

    def __init__(self):
        self.embedding = unit_vector(1)

    def generate_embedding(self, face_image):
        return self.embedding

    def generate_student_representation(self, face_images):
        return self.embedding


def make_png(seed: int = 0, size: int = 200) -> bytes:
    """A noisy image: passes the brightness/blur quality checks of Module 3."""
    rng = np.random.RandomState(seed)
    img = rng.randint(60, 200, size=(size, size, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def image_files(n: int, field: str = "images"):
    return [(field, (f"face{i}.png", make_png(i), "image/png")) for i in range(n)]


FORM = {
    "student_id": "cse1025",
    "name": "Kunal Tiwari",
    "department": "Computer Science",
    "semester": "3",
}


# ---------------------------------------------------------------- fixtures
@pytest.fixture
def engine():
    return create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,  # one shared in-memory DB across threads
    )


@pytest.fixture
def generator():
    return FakeGenerator()


@pytest.fixture
def client(engine, generator):
    matcher = FaceMatcher(cosine_threshold=0.70, generator=generator)
    app = create_app(engine=engine, matcher=matcher)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def one_face(monkeypatch):
    """Pretend the Haar cascade finds exactly one face in every image."""
    fake = lambda image: np.array([[20, 20, 120, 120]])
    monkeypatch.setattr(students_router, "detect_faces", fake)
    monkeypatch.setattr(face_matcher_module, "detect_faces", fake)


@pytest.fixture
def enrolled(client, generator, one_face):
    """A client with student CSE1025 enrolled using embedding unit_vector(1)."""
    r = client.post("/students/register-with-faces", data=FORM, files=image_files(4))
    assert r.status_code == 201, r.text
    return client


# ------------------------------------------------------------------ health
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["database"] == "connected"
    assert body["registered_faces"] == 0
    assert body["match_threshold"] == 0.70


# ---------------------------------------------------------------- students
def test_create_student(client):
    r = client.post("/students", json={
        "student_id": " cse1025 ", "name": "Kunal Tiwari",
        "department": "Computer Science", "semester": 3,
    })
    assert r.status_code == 201
    body = r.json()
    assert body["student_id"] == "CSE1025"  # normalized
    assert body["has_embedding"] is False


def test_create_duplicate_student_conflict(client):
    payload = {"student_id": "CSE1", "name": "Aman Verma", "department": "CS", "semester": 2}
    assert client.post("/students", json=payload).status_code == 201
    assert client.post("/students", json=payload).status_code == 409


@pytest.mark.parametrize("override", [
    {"semester": 99},                 # out of range (schema)
    {"semester": "three"},            # wrong type (schema)
    {"name": "12345"},                # Module 3 rule: name needs letters
    {"student_id": "bad id!"},        # Module 3 rule: ID characters
])
def test_create_student_validation(client, override):
    payload = {"student_id": "CSE1025", "name": "Kunal Tiwari", "department": "CS", "semester": 3}
    payload.update(override)
    assert client.post("/students", json=payload).status_code == 422


def test_list_get_update_delete(client):
    for sid, dept in [("A100", "CS"), ("B200", "EE")]:
        client.post("/students", json={"student_id": sid, "name": "Test Person", "department": dept, "semester": 1})

    assert [s["student_id"] for s in client.get("/students").json()] == ["A100", "B200"]
    assert [s["student_id"] for s in client.get("/students", params={"department": "EE"}).json()] == ["B200"]
    assert len(client.get("/students", params={"limit": 1}).json()) == 1

    assert client.get("/students/a100").json()["department"] == "CS"
    assert client.get("/students/NOPE").status_code == 404

    r = client.put("/students/A100", json={"name": "New Name", "department": "IT", "semester": 5})
    assert r.status_code == 200 and r.json()["name"] == "New Name" and r.json()["semester"] == 5
    assert client.put("/students/NOPE", json={"name": "New Name", "department": "IT", "semester": 5}).status_code == 404

    assert client.delete("/students/A100").status_code == 204
    assert client.get("/students/A100").status_code == 404
    assert client.delete("/students/A100").status_code == 404


def test_register_with_faces(client, generator, one_face):
    r = client.post("/students/register-with-faces", data=FORM, files=image_files(4))
    assert r.status_code == 201
    body = r.json()
    assert body["student"]["student_id"] == "CSE1025"
    assert body["student"]["has_embedding"] is True
    assert body["samples_used"] == 4 and body["re_enrolled"] is False
    assert client.get("/health").json()["registered_faces"] == 1

    # Registering again re-enrolls instead of failing
    again = client.post("/students/register-with-faces", data=FORM, files=image_files(3)).json()
    assert again["re_enrolled"] is True
    assert client.get("/health").json()["registered_faces"] == 1


def test_register_with_faces_too_few_samples(client, one_face):
    r = client.post("/students/register-with-faces", data=FORM, files=image_files(2))
    assert r.status_code == 422
    assert client.get("/students/CSE1025").status_code == 404  # nothing saved


def test_register_with_faces_rejects_bad_samples(client, monkeypatch):
    monkeypatch.setattr(students_router, "detect_faces", lambda image: np.empty((0, 4), dtype=int))
    files = image_files(3) + [("images", ("broken.png", b"not an image", "image/png"))]
    r = client.post("/students/register-with-faces", data=FORM, files=files)
    assert r.status_code == 422
    reasons = [s["reason"] for s in r.json()["detail"]["rejected_samples"]]
    assert "No face detected." in reasons
    assert any("not a valid image" in x for x in reasons)


def test_register_with_faces_rejects_multiple_faces(client, monkeypatch):
    monkeypatch.setattr(students_router, "detect_faces",
                        lambda image: np.array([[0, 0, 80, 80], [100, 100, 80, 80]]))
    r = client.post("/students/register-with-faces", data=FORM, files=image_files(3))
    assert r.status_code == 422
    assert "2 faces" in r.json()["detail"]["rejected_samples"][0]["reason"]


# ------------------------------------------------------ recognition: frame
def test_recognize_frame_marks_attendance_once(enrolled, generator):
    generator.embedding = unit_vector(1)  # same face as enrolled

    first = enrolled.post("/recognition/frame", files=[("image", ("f.png", make_png(9), "image/png"))])
    assert first.status_code == 200
    body = first.json()
    assert body["faces_detected"] == 1 and body["known_faces"] == 1
    face = body["results"][0]
    assert face["student_id"] == "CSE1025" and face["is_known"] is True
    assert face["attendance"]["marked"] is True
    assert face["bbox"] == [20, 20, 120, 120]

    second = enrolled.post("/recognition/frame", files=[("image", ("f.png", make_png(9), "image/png"))]).json()
    att = second["results"][0]["attendance"]
    assert att["marked"] is False and att["already_marked"] is True

    assert len(enrolled.get("/attendance/daily").json()) == 1  # still one record


def test_recognize_frame_unknown_face_not_marked(enrolled, generator):
    generator.embedding = unit_vector(2)  # a stranger
    body = enrolled.post("/recognition/frame", files=[("image", ("f.png", make_png(3), "image/png"))]).json()
    assert body["unknown_faces"] == 1
    assert body["results"][0]["is_known"] is False
    assert body["results"][0]["attendance"] is None
    assert enrolled.get("/attendance/daily").json() == []


def test_recognize_frame_without_marking(enrolled, generator):
    generator.embedding = unit_vector(1)
    r = enrolled.post("/recognition/frame",
                      files=[("image", ("f.png", make_png(4), "image/png"))],
                      data={"mark_attendance": "false"}).json()
    assert r["known_faces"] == 1 and r["results"][0]["attendance"] is None
    assert enrolled.get("/attendance/daily").json() == []


def test_recognize_frame_no_face(client, monkeypatch):
    monkeypatch.setattr(face_matcher_module, "detect_faces", lambda image: np.empty((0, 4), dtype=int))
    r = client.post("/recognition/frame", files=[("image", ("f.png", make_png(), "image/png"))]).json()
    assert r == {"faces_detected": 0, "known_faces": 0, "unknown_faces": 0, "results": []}


def test_recognize_frame_invalid_image(client):
    r = client.post("/recognition/frame", files=[("image", ("f.txt", b"hello", "text/plain"))])
    assert r.status_code == 400


# --------------------------------------------------- recognition: embedding
def test_match_embedding_known_and_unknown(enrolled):
    known = enrolled.post("/recognition/embedding", json={"embedding": unit_vector(1).tolist(), "mark_attendance": True})
    assert known.status_code == 200
    assert known.json()["student_id"] == "CSE1025"
    assert known.json()["attendance"]["marked"] is True

    stranger = enrolled.post("/recognition/embedding", json={"embedding": unit_vector(5).tolist(), "mark_attendance": True}).json()
    assert stranger["is_known"] is False and stranger["attendance"] is None


def test_match_embedding_validation(client):
    assert client.post("/recognition/embedding", json={"embedding": [0.1] * 10}).status_code == 422
    assert client.post("/recognition/embedding", json={"embedding": [0.0] * 512}).status_code == 422


def test_match_embedding_empty_gallery_is_unknown(client):
    r = client.post("/recognition/embedding", json={"embedding": unit_vector(1).tolist()}).json()
    assert r["is_known"] is False


# -------------------------------------------------------------- attendance
def _make_student(client, sid="CSE1025", dept="Computer Science"):
    client.post("/students", json={"student_id": sid, "name": "Kunal Tiwari", "department": dept, "semester": 3})


def test_manual_mark_attendance(client):
    _make_student(client)
    r = client.post("/attendance/mark", json={"student_id": "cse1025"})
    assert r.status_code == 201
    assert r.json()["status"] == "Present" and r.json()["student_name"] == "Kunal Tiwari"

    assert client.post("/attendance/mark", json={"student_id": "CSE1025"}).status_code == 409
    assert client.post("/attendance/mark", json={"student_id": "GHOST1"}).status_code == 404
    assert client.post("/attendance/mark", json={"student_id": "CSE1025", "status": "Maybe"}).status_code == 422


def test_daily_attendance_filters(client):
    _make_student(client, "CSE1", "Computer Science")
    _make_student(client, "EE1", "Electrical")
    client.post("/attendance/mark", json={"student_id": "CSE1"})
    client.post("/attendance/mark", json={"student_id": "EE1"})

    assert len(client.get("/attendance/daily").json()) == 2
    cs = client.get("/attendance/daily", params={"department": "Computer Science"}).json()
    assert [r["student_id"] for r in cs] == ["CSE1"]
    assert client.get("/attendance/daily", params={"date": "2020-01-01"}).json() == []
    assert client.get("/attendance/daily", params={"date": "not-a-date"}).status_code == 422


def test_student_history(client, engine):
    _make_student(client)
    mgr = AttendanceManager(engine=engine)
    today = date.today()
    for days_ago in (0, 1, 2):
        mgr.mark_attendance("CSE1025", target_date=today - timedelta(days=days_ago))

    history = client.get("/attendance/student/cse1025").json()
    assert [h["date"] for h in history] == [str(today - timedelta(days=d)) for d in (0, 1, 2)]  # newest first

    ranged = client.get("/attendance/student/CSE1025", params={"start_date": str(today - timedelta(days=1))}).json()
    assert len(ranged) == 2

    assert client.get("/attendance/student/GHOST1").status_code == 404
    bad = client.get("/attendance/student/CSE1025", params={"start_date": "2026-02-01", "end_date": "2026-01-01"})
    assert bad.status_code == 422


def test_monthly_summary(client, engine):
    _make_student(client, "CSE1")
    _make_student(client, "CSE2")
    mgr = AttendanceManager(engine=engine)
    mgr.mark_attendance("CSE1", target_date=date(2026, 8, 3))
    mgr.mark_attendance("CSE1", target_date=date(2026, 8, 4))
    mgr.mark_attendance("CSE2", target_date=date(2026, 8, 3))

    rows = {r["student_id"]: r for r in client.get("/attendance/monthly", params={"year": 2026, "month": 8}).json()}
    assert rows["CSE1"]["present_days"] == 2 and rows["CSE1"]["attendance_percentage"] == 100.0
    assert rows["CSE2"]["present_days"] == 1 and rows["CSE2"]["attendance_percentage"] == 50.0
    assert rows["CSE1"]["total_active_days"] == 2

    assert client.get("/attendance/monthly", params={"year": 2026, "month": 13}).status_code == 422
    assert client.get("/attendance/monthly", params={"year": 2026}).status_code == 422


def test_delete_student_removes_attendance_and_face(enrolled, generator):
    enrolled.post("/attendance/mark", json={"student_id": "CSE1025"})
    assert enrolled.delete("/students/CSE1025").status_code == 204
    assert enrolled.get("/attendance/daily").json() == []
    assert enrolled.get("/health").json()["registered_faces"] == 0
    # deleted student's face is no longer recognized
    r = enrolled.post("/recognition/embedding", json={"embedding": unit_vector(1).tolist()}).json()
    assert r["is_known"] is False


# ------------------------------------------------------------- persistence
def test_gallery_reloaded_from_database_on_startup(engine, generator, one_face):
    first = create_app(engine=engine, matcher=FaceMatcher(generator=generator))
    with TestClient(first) as c:
        assert c.post("/students/register-with-faces", data=FORM, files=image_files(3)).status_code == 201

    # "Restart" the server: brand-new app + matcher, same database
    second = create_app(engine=engine, matcher=FaceMatcher(generator=generator))
    with TestClient(second) as c:
        assert c.get("/health").json()["registered_faces"] == 1
        r = c.post("/recognition/embedding", json={"embedding": unit_vector(1).tolist()}).json()
        assert r["student_id"] == "CSE1025"


def test_openapi_docs_available(client):
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    for p in ["/health", "/students", "/recognition/frame", "/attendance/daily", "/attendance/mark"]:
        assert p in paths
