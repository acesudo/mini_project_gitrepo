"""
Module 6 — Attendance Database Layer

Provides SQLAlchemy declarative models, safe engine initialization, transaction
management, duplicate attendance prevention, and daily/monthly reporting.
"""

from .models import Base, Student, Attendance
from .database import (
    get_engine,
    get_db_session,
    init_db,
    get_database_url,
    SessionLocal,
)
from .attendance_manager import AttendanceManager

__all__ = [
    "Base",
    "Student",
    "Attendance",
    "get_engine",
    "get_db_session",
    "init_db",
    "get_database_url",
    "SessionLocal",
    "AttendanceManager",
]
