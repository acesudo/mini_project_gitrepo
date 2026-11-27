"""
Module 9 — FastAPI Authentication and Role Dependencies.
"""

from datetime import datetime, timezone
from typing import List, Optional
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

from module9.config import config
from module9.security import decode_access_token
from module9.models import User
from module9.user_manager import UserManager

# HTTP Bearer scheme (handles Authorization: Bearer <token>)
security_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
) -> Optional[User]:
    """
    Extracts and validates the JWT bearer token.
    If AUTH_ENABLED is False and no token is provided, returns a synthetic admin user
    so tests and dev mode remain friction-free.
    When AUTH_ENABLED is True, strictly validates token and loads User from database.
    """
    manager = request.app.state.core.manager
    with manager.get_session() as session:
        user_mgr = UserManager(session)

        # Token provided in request
        if credentials and credentials.credentials:
            token = credentials.credentials
            try:
                payload = decode_access_token(token)
                username: str = payload.get("sub")
                if not username:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid token claims: missing subject.",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
            except jwt.ExpiredSignatureError:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Access token has expired.",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            except jwt.PyJWTError:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Could not validate credentials.",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            user = user_mgr.get_by_username(username)
            if not user or not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User inactive or no longer exists.",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            # Detach user or construct an in-memory replica with attributes
            detached_user = User(
                id=user.id,
                username=user.username,
                hashed_password=user.hashed_password,
                role=user.role,
                full_name=user.full_name,
                is_active=user.is_active,
                created_at=user.created_at,
            )
            return detached_user

        # No token provided
        if not config.AUTH_ENABLED:
            # Development/Testing bypass
            return User(
                id=0,
                username="dev_admin",
                role="admin",
                full_name="Developer Admin",
                is_active=True,
                created_at=datetime.now(timezone.utc),
            )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required: missing Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


class RoleChecker:
    """
    FastAPI dependency for Role-Based Access Control (RBAC).
    Usage: Depends(RoleChecker(["admin", "teacher"]))
    """

    def __init__(self, allowed_roles: List[str]):
        self.allowed_roles = set(allowed_roles)

    def __call__(self, current_user: User = Depends(get_current_user)) -> User:
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required.",
            )
        if current_user.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access forbidden: role '{current_user.role}' lacks permission. Allowed: {sorted(self.allowed_roles)}",
            )
        return current_user
