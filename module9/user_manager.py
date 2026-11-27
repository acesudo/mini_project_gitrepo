"""
Module 9 — User Manager for Authentication and Role Administration.
"""

from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select

from module9.models import User
from module9.security import hash_password, verify_password
from module9.config import config


class UserManager:
    """
    CRUD and credential operations for User records.
    """

    def __init__(self, session: Session):
        self.session = session

    def seed_initial_admin(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> Optional[User]:
        """
        Creates the primary administrator account if no admin user exists in DB.
        """
        admin_uname = username or config.INITIAL_ADMIN_USERNAME
        admin_pwd = password or config.INITIAL_ADMIN_PASSWORD

        existing = self.get_by_username(admin_uname)
        if existing:
            return existing

        user = User(
            username=admin_uname,
            hashed_password=hash_password(admin_pwd),
            role="admin",
            full_name="System Administrator",
            is_active=True,
        )
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)
        return user

    def get_by_username(self, username: str) -> Optional[User]:
        stmt = select(User).where(User.username == username)
        return self.session.execute(stmt).scalar_one_or_none()

    def get_by_id(self, user_id: int) -> Optional[User]:
        stmt = select(User).where(User.id == user_id)
        return self.session.execute(stmt).scalar_one_or_none()

    def authenticate_user(self, username: str, plain_password: str) -> Optional[User]:
        user = self.get_by_username(username)
        if not user or not user.is_active:
            return None
        if not verify_password(plain_password, user.hashed_password):
            return None
        return user

    def create_user(
        self,
        username: str,
        password: str,
        role: str = "teacher",
        full_name: Optional[str] = None,
    ) -> User:
        if self.get_by_username(username):
            raise ValueError(f"Username '{username}' already exists.")
        user = User(
            username=username,
            hashed_password=hash_password(password),
            role=role,
            full_name=full_name,
            is_active=True,
        )
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)
        return user

    def list_users(self) -> List[User]:
        stmt = select(User).order_by(User.id)
        return list(self.session.execute(stmt).scalars().all())
