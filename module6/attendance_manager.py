"""
Module 6 — Attendance Manager & Database Repository

Encapsulates CRUD operations, duplicate attendance prevention, daily/monthly
attendance reporting, and database-backed embedding retrieval for Module 5.
"""

import sys
from pathlib import Path
from datetime import date, datetime, time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sqlalchemy import func, extract, and_
from sqlalchemy.orm import Session

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from module6.models import Student, Attendance
from module6.database import get_db_session, init_db
from module4.face_embedding import FaceEmbeddingGenerator


class AttendanceManager:
    """
    Attendance and student management service.
    """

    def __init__(self, engine=None) -> None:
        self.engine = engine
        init_db(self.engine)

    def mark_attendance(
        self,
        student_id: str,
        target_date: Optional[date] = None,
        target_time: Optional[time] = None,
        status: str = "Present",
        confidence: Optional[float] = None,
        session: Optional[Session] = None,
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Record attendance for a student, enforcing single-daily-entry rules.

        Returns:
            (success: bool, message: str, record: Optional[dict])
        """
        sid = str(student_id).strip().upper()
        rec_date = target_date or date.today()
        rec_time = target_time or datetime.now().time()

        def _execute(sess: Session):
            # 1. Verify student exists in registry
            student = sess.query(Student).filter(Student.student_id == sid).first()
            if not student:
                return False, f"Student '{sid}' is not registered in the system.", None

            # 2. Duplicate Attendance Check
            existing = (
                sess.query(Attendance)
                .filter(Attendance.student_id == sid, Attendance.date == rec_date)
                .first()
            )
            if existing:
                return (
                    False,
                    f"Attendance already marked for student '{sid}' on {rec_date}.",
                    existing.to_dict(),
                )

            # 3. Create attendance record
            new_record = Attendance(
                student_id=sid,
                date=rec_date,
                time=rec_time,
                status=status,
                confidence=confidence,
            )
            sess.add(new_record)
            sess.flush()  # Populate generated primary key

            return (
                True,
                f"Attendance marked successfully for student '{sid}' on {rec_date}.",
                new_record.to_dict(),
            )

        if session:
            return _execute(session)
        else:
            with get_db_session(self.engine) as sess:
                return _execute(sess)

    def get_daily_attendance(
        self,
        target_date: Optional[date] = None,
        department: Optional[str] = None,
        session: Optional[Session] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all attendance records for a specific date (defaults to today).
        """
        rec_date = target_date or date.today()

        def _execute(sess: Session):
            query = (
                sess.query(Attendance)
                .join(Student, Attendance.student_id == Student.student_id)
                .filter(Attendance.date == rec_date)
            )
            if department:
                query = query.filter(Student.department == department)

            records = query.order_by(Attendance.time.asc()).all()
            return [rec.to_dict() for rec in records]

        if session:
            return _execute(session)
        else:
            with get_db_session(self.engine) as sess:
                return _execute(sess)

    def get_student_attendance_history(
        self,
        student_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        session: Optional[Session] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chronological attendance history for a single student.
        """
        sid = str(student_id).strip().upper()

        def _execute(sess: Session):
            query = sess.query(Attendance).filter(Attendance.student_id == sid)
            if start_date:
                query = query.filter(Attendance.date >= start_date)
            if end_date:
                query = query.filter(Attendance.date <= end_date)

            records = query.order_by(Attendance.date.desc(), Attendance.time.desc()).all()
            return [rec.to_dict() for rec in records]

        if session:
            return _execute(session)
        else:
            with get_db_session(self.engine) as sess:
                return _execute(sess)

    def get_monthly_attendance_summary(
        self,
        year: int,
        month: int,
        department: Optional[str] = None,
        session: Optional[Session] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate monthly attendance summary with attendance counts and percentages per student.
        """
        def _execute(sess: Session):
            # 1. Total unique attendance dates in the month (active school days)
            total_days_subq = (
                sess.query(func.count(func.distinct(Attendance.date)))
                .filter(
                    extract("year", Attendance.date) == year,
                    extract("month", Attendance.date) == month,
                )
                .scalar()
                or 0
            )

            # 2. Get students filter
            stu_query = sess.query(Student)
            if department:
                stu_query = stu_query.filter(Student.department == department)
            students = stu_query.order_by(Student.student_id).all()

            summary = []
            for stu in students:
                present_days = (
                    sess.query(func.count(Attendance.attendance_id))
                    .filter(
                        Attendance.student_id == stu.student_id,
                        Attendance.status == "Present",
                        extract("year", Attendance.date) == year,
                        extract("month", Attendance.date) == month,
                    )
                    .scalar()
                    or 0
                )

                percentage = (
                    round((present_days / total_days_subq) * 100.0, 2)
                    if total_days_subq > 0
                    else 0.0
                )

                summary.append({
                    "student_id": stu.student_id,
                    "name": stu.name,
                    "department": stu.department,
                    "semester": stu.semester,
                    "month": month,
                    "year": year,
                    "total_active_days": total_days_subq,
                    "present_days": present_days,
                    "attendance_percentage": percentage,
                })

            return summary

        if session:
            return _execute(session)
        else:
            with get_db_session(self.engine) as sess:
                return _execute(sess)

    def register_student(
        self,
        student_id: str,
        name: str,
        department: str,
        semester: int,
        embedding: Optional[np.ndarray] = None,
        session: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """
        Register a new student or update their face embedding.
        """
        sid = str(student_id).strip().upper()
        binary_emb = (
            FaceEmbeddingGenerator.embedding_to_bytes(embedding)
            if embedding is not None
            else None
        )

        def _execute(sess: Session):
            student = sess.query(Student).filter(Student.student_id == sid).first()
            if student:
                student.name = name.strip()
                student.department = department.strip()
                student.semester = semester
                if binary_emb:
                    student.face_embedding = binary_emb
            else:
                student = Student(
                    student_id=sid,
                    name=name.strip(),
                    department=department.strip(),
                    semester=semester,
                    face_embedding=binary_emb,
                )
                sess.add(student)

            sess.flush()
            return student.to_dict()

        if session:
            return _execute(session)
        else:
            with get_db_session(self.engine) as sess:
                return _execute(sess)

    def load_all_registered_embeddings(
        self, session: Optional[Session] = None
    ) -> Dict[str, np.ndarray]:
        """
        Load all stored face embeddings from database into a {student_id: np.ndarray} dictionary.
        Directly populates Module 5 FaceMatcher.
        """
        def _execute(sess: Session):
            students = (
                sess.query(Student.student_id, Student.face_embedding)
                .filter(Student.face_embedding.isnot(None))
                .all()
            )
            embeddings = {}
            for sid, raw_bytes in students:
                try:
                    embeddings[sid] = FaceEmbeddingGenerator.bytes_to_embedding(raw_bytes)
                except Exception as err:
                    print(f"Warning: Failed to deserialize embedding for student {sid}: {err}")
            return embeddings

        if session:
            return _execute(session)
        else:
            with get_db_session(self.engine) as sess:
                return _execute(sess)


def main():
    """Demonstrate Module 6 database functionality."""
    print("=" * 60)
    print("MODULE 6 — Attendance Database Demo")
    print("=" * 60)

    mgr = AttendanceManager()
    print("Database initialized successfully!")

    # 1. Register a test student with a 512-d unit embedding
    np.random.seed(42)
    fake_emb = np.random.randn(512).astype(np.float32)
    fake_emb /= np.linalg.norm(fake_emb)

    stu = mgr.register_student(
        student_id="DEMO_CSE01",
        name="Aarav Sharma",
        department="Computer Science",
        semester=4,
        embedding=fake_emb,
    )
    print(f"\n--- 1. Registered Student: {stu['name']} ({stu['student_id']}) ---")

    # 2. Mark attendance for today
    success, msg, rec = mgr.mark_attendance(
        student_id="DEMO_CSE01",
        confidence=0.942,
    )
    print(f"\n--- 2. First Attendance Entry ---")
    print(f"Success: {success}")
    print(f"Message: {msg}")

    # 3. Test Duplicate Attendance Protection
    dup_success, dup_msg, dup_rec = mgr.mark_attendance(
        student_id="DEMO_CSE01",
        confidence=0.915,
    )
    print(f"\n--- 3. Duplicate Attendance Entry Attempt ---")
    print(f"Duplicate Blocked (Expected False): {dup_success}")
    print(f"Message: {dup_msg}")

    # 4. Query Daily Attendance
    daily = mgr.get_daily_attendance()
    print(f"\n--- 4. Today's Attendance List ({len(daily)} students) ---")
    for row in daily:
        print(f"  [{row['time']}] {row['student_id']} - {row['student_name']} ({row['status']}, confidence: {row['confidence']})")

    # 5. Load Registered Embeddings into Memory
    gallery = mgr.load_all_registered_embeddings()
    print(f"\n--- 5. Loaded {len(gallery)} Biometric Embeddings for FaceMatcher ---")
    if "DEMO_CSE01" in gallery:
        print(f"  DEMO_CSE01 Embedding Shape: {gallery['DEMO_CSE01'].shape}, Norm: {np.linalg.norm(gallery['DEMO_CSE01']):.4f}")

    print("\nModule 6 demonstration completed successfully.")


if __name__ == "__main__":
    main()
