"""
Module 6 — SQLAlchemy Declarative Models

Defines the relational schema for Students and Attendance according to database.md.
Enforces foreign keys, timestamps, cascades, and duplicate attendance prevention.
"""

from datetime import date, datetime, time
from typing import Dict, Any, Optional
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Date,
    Time,
    DateTime,
    LargeBinary,
    ForeignKey,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Student(Base):
    """
    Student model representing registered students and their biometric face representation.
    """
    __tablename__ = "students"

    student_id = Column(String(30), primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    department = Column(String(100), nullable=False)
    semester = Column(Integer, nullable=False)
    face_embedding = Column(LargeBinary, nullable=True)  # 2048 raw bytes for 512 float32 embedding
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # One-to-many relationship: One student has multiple attendance records
    attendances = relationship(
        "Attendance",
        back_populates="student",
        cascade="all, delete-orphan",
        order_by="desc(Attendance.date), desc(Attendance.time)",
    )

    def to_dict(self, include_embedding: bool = False) -> Dict[str, Any]:
        result = {
            "student_id": self.student_id,
            "name": self.name,
            "department": self.department,
            "semester": self.semester,
            "has_embedding": self.face_embedding is not None,
            "created_at": str(self.created_at) if self.created_at else None,
        }
        if include_embedding and self.face_embedding:
            result["face_embedding"] = self.face_embedding
        return result


class Attendance(Base):
    """
    Attendance model representing classroom attendance records.
    Enforces duplicate prevention via UniqueConstraint(student_id, date).
    """
    __tablename__ = "attendance"

    attendance_id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(
        String(30),
        ForeignKey("students.student_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    date = Column(Date, nullable=False, index=True, default=date.today)
    time = Column(Time, nullable=False, default=lambda: datetime.now().time())
    status = Column(String(20), nullable=False, default="Present")
    confidence = Column(Float, nullable=True)  # Recognition confidence score (0.0 to 1.0)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Many-to-one relationship back to Student
    student = relationship("Student", back_populates="attendances")

    # Enforce duplicate attendance protection at the database level
    __table_args__ = (
        UniqueConstraint("student_id", "date", name="uq_student_daily_attendance"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attendance_id": self.attendance_id,
            "student_id": self.student_id,
            "student_name": self.student.name if self.student else None,
            "department": self.student.department if self.student else None,
            "semester": self.student.semester if self.student else None,
            "date": str(self.date),
            "time": self.time.strftime("%H:%M:%S") if self.time else None,
            "status": self.status,
            "confidence": round(float(self.confidence), 4) if self.confidence is not None else None,
            "created_at": str(self.created_at) if self.created_at else None,
        }
