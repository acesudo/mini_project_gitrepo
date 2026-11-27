"""
Module 7 — Small database helpers

Module 6's AttendanceManager covers registration, attendance and reports. The API
also needs to list, fetch and delete students, so those queries live here and
reuse Module 6's models and transactional session.
"""

from typing import Any, Dict, List, Optional

from module6.database import get_db_session
from module6.models import Student


def student_exists(manager, student_id: str) -> bool:
    return get_student(manager, student_id) is not None


def get_student(manager, student_id: str) -> Optional[Dict[str, Any]]:
    sid = student_id.strip().upper()
    with get_db_session(manager.engine) as sess:
        student = sess.get(Student, sid)
        return student.to_dict() if student else None


def list_students(
    manager,
    department: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    with get_db_session(manager.engine) as sess:
        query = sess.query(Student)
        if department:
            query = query.filter(Student.department == department)
        rows = query.order_by(Student.student_id).offset(offset).limit(limit).all()
        return [s.to_dict() for s in rows]


def delete_student(manager, student_id: str) -> bool:
    """Delete a student; their attendance rows are removed by the cascade rule."""
    sid = student_id.strip().upper()
    with get_db_session(manager.engine) as sess:
        student = sess.get(Student, sid)
        if student is None:
            return False
        sess.delete(student)
        return True
