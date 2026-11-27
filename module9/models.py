"""
Module 9 — User Database Model for Role-Based Access Control (RBAC).
"""

from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, Integer, String
from module6.models import Base


class User(Base):
    """
    User entity for administrative access and API authentication.
    Supported roles:
      - 'admin': Full system access (students, attendance, users, settings)
      - 'teacher': Read attendance, history, mark manual attendance, monthly summaries
      - 'kiosk': Machine account for live attendance terminal to post recognition frames
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default="teacher")  # admin | teacher | kiosk
    full_name = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    def __repr__(self) -> str:
        return f"<User(id={self.id}, username='{self.username}', role='{self.role}')>"
