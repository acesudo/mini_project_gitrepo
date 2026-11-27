"""
Module 9 — Authentication and User Management Router.
Prefix: /api/v1/auth
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, Request, status

from module9.schemas import LoginRequest, TokenResponse, UserCreate, UserResponse
from module9.dependencies import get_current_user, RoleChecker
from module9.user_manager import UserManager
from module9.security import create_access_token
from module9.models import User

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication & Security"])


@router.post("/login", response_model=TokenResponse)
def login(request: Request, payload: LoginRequest):
    """
    Authenticate with username and password to receive a JWT Bearer access token.
    """
    manager = request.app.state.core.manager
    with manager.get_session() as session:
        user_mgr = UserManager(session)
        user = user_mgr.authenticate_user(payload.username, payload.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = create_access_token(
            data={"sub": user.username, "role": user.role, "uid": user.id}
        )
        return TokenResponse(
            access_token=token,
            expires_in_minutes=60,
            user_id=user.id,
            username=user.username,
            role=user.role,
        )


@router.get("/me", response_model=UserResponse)
def get_authenticated_profile(current_user: User = Depends(get_current_user)):
    """
    Retrieve details and role permissions of the currently authenticated user.
    """
    return UserResponse.model_validate(current_user)


@router.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_system_user(
    request: Request,
    payload: UserCreate,
    current_user: User = Depends(RoleChecker(["admin"])),
):
    """
    Create a new system user (admin, teacher, or kiosk). Requires 'admin' privileges.
    """
    manager = request.app.state.core.manager
    with manager.get_session() as session:
        user_mgr = UserManager(session)
        try:
            new_user = user_mgr.create_user(
                username=payload.username,
                password=payload.password,
                role=payload.role,
                full_name=payload.full_name,
            )
            return UserResponse.model_validate(new_user)
        except ValueError as err:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(err))


@router.get("/users", response_model=List[UserResponse])
def list_system_users(
    request: Request,
    current_user: User = Depends(RoleChecker(["admin"])),
):
    """
    List all system accounts. Requires 'admin' privileges.
    """
    manager = request.app.state.core.manager
    with manager.get_session() as session:
        user_mgr = UserManager(session)
        return [UserResponse.model_validate(u) for u in user_mgr.list_users()]
