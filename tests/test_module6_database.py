"""
Unit and integration tests for Module 6 — Attendance Database.
Tested against isolated in-memory databases to ensure zero side-effects.
"""

from datetime import date, datetime, timedelta, time
import pytest
import numpy as np
from sqlalchemy.exc import IntegrityError

from module6.models import Base, Student, Attendance
from module6.database import get_engine, get_db_session
from module6.attendance_manager import AttendanceManager


@pytest.fixture
def test_engine():
    """Provides a fresh isolated in-memory SQLite engine for each test."""
    engine = get_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture
def manager(test_engine):
    """Provides an AttendanceManager bound to the test engine."""
    return AttendanceManager(engine=test_engine)


@pytest.fixture
def sample_embedding():
    """Generates a reproducible 512-d unit vector."""
    np.random.seed(42)
    v = np.random.randn(512).astype(np.float32)
    return v / np.linalg.norm(v)


def test_schema_creation(test_engine):
    """Verify tables students and attendance exist with proper columns."""
    from sqlalchemy import inspect
    inspector = inspect(test_engine)
    tables = inspector.get_table_names()

    assert "students" in tables
    assert "attendance" in tables

    student_cols = [c["name"] for c in inspector.get_columns("students")]
    assert "student_id" in student_cols
    assert "name" in student_cols
    assert "face_embedding" in student_cols

    attendance_cols = [c["name"] for c in inspector.get_columns("attendance")]
    assert "attendance_id" in attendance_cols
    assert "student_id" in attendance_cols
    assert "date" in attendance_cols
    assert "time" in attendance_cols
    assert "confidence" in attendance_cols


def test_student_registration_with_embedding(manager, sample_embedding):
    """Verify student registration and embedding storage."""
    student_dict = manager.register_student(
        student_id="CSE101",
        name="John Doe",
        department="Computer Science",
        semester=3,
        embedding=sample_embedding,
    )

    assert student_dict["student_id"] == "CSE101"
    assert student_dict["name"] == "John Doe"
    assert student_dict["has_embedding"] is True

    # Verify retrieval
    gallery = manager.load_all_registered_embeddings()
    assert "CSE101" in gallery
    assert gallery["CSE101"].shape == (512,)
    assert np.allclose(gallery["CSE101"], sample_embedding, atol=1e-5)


def test_mark_attendance_success(manager, sample_embedding):
    """Verify marking attendance for a registered student."""
    manager.register_student("CSE101", "John Doe", "CS", 3, sample_embedding)

    success, msg, rec = manager.mark_attendance(
        student_id="CSE101",
        target_date=date(2026, 11, 25),
        confidence=0.925,
    )

    assert success is True
    assert "successfully" in msg.lower()
    assert rec["student_id"] == "CSE101"
    assert rec["date"] == "2026-11-25"
    assert rec["confidence"] == 0.925
    assert rec["status"] == "Present"


def test_mark_attendance_unregistered_student(manager):
    """Attempting to mark attendance for an unregistered student must fail."""
    success, msg, rec = manager.mark_attendance("STRANGER_01")
    assert success is False
    assert "not registered" in msg.lower()
    assert rec is None


def test_duplicate_attendance_prevention(manager, sample_embedding):
    """A second attendance attempt on the same day must be rejected."""
    manager.register_student("CSE101", "John Doe", "CS", 3, sample_embedding)
    today = date(2026, 11, 25)

    # 1. First attempt - Success
    ok1, msg1, rec1 = manager.mark_attendance("CSE101", target_date=today, confidence=0.91)
    assert ok1 is True

    # 2. Duplicate attempt on same date - Blocked
    ok2, msg2, rec2 = manager.mark_attendance("CSE101", target_date=today, confidence=0.88)
    assert ok2 is False
    assert "already marked" in msg2.lower()
    assert rec2["attendance_id"] == rec1["attendance_id"]


def test_unique_constraint_enforced_at_db_level(test_engine, manager, sample_embedding):
    """Direct database insert of duplicate (student_id, date) must raise IntegrityError."""
    manager.register_student("CSE101", "John Doe", "CS", 3, sample_embedding)
    today = date(2026, 11, 25)

    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=test_engine)
    sess = Session()
    try:
        sess.add(Attendance(student_id="CSE101", date=today, status="Present"))
        sess.commit()

        # Second raw insert with identical (student_id, date) must trigger database unique constraint
        sess.add(Attendance(student_id="CSE101", date=today, status="Present"))
        with pytest.raises(IntegrityError):
            sess.commit()
    finally:
        sess.rollback()
        sess.close()


def test_different_dates_allowed(manager, sample_embedding):
    """The same student can be marked present on multiple distinct dates."""
    manager.register_student("CSE101", "John Doe", "CS", 3, sample_embedding)

    day1 = date(2026, 11, 24)
    day2 = date(2026, 11, 25)

    ok1, _, _ = manager.mark_attendance("CSE101", target_date=day1)
    ok2, _, _ = manager.mark_attendance("CSE101", target_date=day2)

    assert ok1 is True
    assert ok2 is True


def test_daily_attendance_query(manager, sample_embedding):
    """Verify daily attendance returns joined student records."""
    manager.register_student("CSE101", "Alice", "CSE", 3, sample_embedding)
    manager.register_student("CSE102", "Bob", "ECE", 5, sample_embedding)

    target = date(2026, 11, 25)
    manager.mark_attendance("CSE101", target_date=target, target_time=time(9, 15))
    manager.mark_attendance("CSE102", target_date=target, target_time=time(9, 20))

    daily = manager.get_daily_attendance(target_date=target)
    assert len(daily) == 2
    assert daily[0]["student_id"] == "CSE101"
    assert daily[0]["student_name"] == "Alice"
    assert daily[1]["student_id"] == "CSE102"
    assert daily[1]["student_name"] == "Bob"

    # Filter by department
    cse_only = manager.get_daily_attendance(target_date=target, department="CSE")
    assert len(cse_only) == 1
    assert cse_only[0]["student_id"] == "CSE101"


def test_student_attendance_history(manager, sample_embedding):
    """Verify student history query returns ordered logs."""
    manager.register_student("CSE101", "Alice", "CSE", 3, sample_embedding)

    d1 = date(2026, 11, 20)
    d2 = date(2026, 11, 22)
    manager.mark_attendance("CSE101", target_date=d1)
    manager.mark_attendance("CSE101", target_date=d2)

    history = manager.get_student_attendance_history("CSE101")
    assert len(history) == 2
    # Ordered descending
    assert history[0]["date"] == "2026-11-22"
    assert history[1]["date"] == "2026-11-20"


def test_monthly_attendance_summary(manager, sample_embedding):
    """Verify monthly summary calculates sessions and percentages correctly."""
    manager.register_student("CSE101", "Alice", "CSE", 3, sample_embedding)
    manager.register_student("CSE102", "Bob", "CSE", 3, sample_embedding)

    # 3 unique school days in November 2026
    d1 = date(2026, 11, 10)
    d2 = date(2026, 11, 11)
    d3 = date(2026, 11, 12)

    # Alice attends 3 days, Bob attends 1 day
    manager.mark_attendance("CSE101", target_date=d1)
    manager.mark_attendance("CSE101", target_date=d2)
    manager.mark_attendance("CSE101", target_date=d3)
    manager.mark_attendance("CSE102", target_date=d1)

    summary = manager.get_monthly_attendance_summary(year=2026, month=11)
    assert len(summary) == 2

    alice_summary = next(s for s in summary if s["student_id"] == "CSE101")
    assert alice_summary["total_active_days"] == 3
    assert alice_summary["present_days"] == 3
    assert alice_summary["attendance_percentage"] == 100.0

    bob_summary = next(s for s in summary if s["student_id"] == "CSE102")
    assert bob_summary["total_active_days"] == 3
    assert bob_summary["present_days"] == 1
    assert bob_summary["attendance_percentage"] == 33.33


def test_cascade_delete(test_engine, manager, sample_embedding):
    """Deleting a student must automatically cascade delete their attendance records."""
    manager.register_student("CSE101", "Alice", "CSE", 3, sample_embedding)
    manager.mark_attendance("CSE101", target_date=date(2026, 11, 25))

    with get_db_session(test_engine) as sess:
        stu = sess.query(Student).filter(Student.student_id == "CSE101").first()
        sess.delete(stu)

    with get_db_session(test_engine) as sess:
        assert sess.query(Student).filter(Student.student_id == "CSE101").first() is None
        assert sess.query(Attendance).filter(Attendance.student_id == "CSE101").count() == 0
