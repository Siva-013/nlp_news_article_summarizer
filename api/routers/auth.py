"""
Authentication Router
POST /api/v1/auth/register  — create an account
POST /api/v1/auth/login     — exchange credentials for a JWT
GET  /api/v1/auth/me        — current authenticated user
"""

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_current_user, get_db
from api.schemas import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from api.security import create_access_token, hash_password, verify_password, TOKEN_EXPIRE_HOURS
from database.notes_history import NotesHistoryManager

router = APIRouter(prefix="/auth", tags=["Authentication"])


def _user_response(user: dict) -> UserResponse:
    return UserResponse(id=user["id"], username=user["username"], created_at=user["created_at"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a New User",
    description="Create a user account with a bcrypt-hashed password.",
)
def register(
    request: RegisterRequest,
    db: NotesHistoryManager = Depends(get_db),
):
    try:
        user_id = db.create_user(request.username, hash_password(request.password))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    user = db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve newly created user.",
        )
    return _user_response(user)


@router.post(
    "/login",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Login",
    description="Verify credentials and return a signed JWT bearer token.",
)
def login(
    request: LoginRequest,
    db: NotesHistoryManager = Depends(get_db),
):
    user = db.get_user_by_username(request.username.strip())
    if not user or not verify_password(request.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(user["id"], user["username"])
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in_hours=TOKEN_EXPIRE_HOURS,
        user=_user_response(user),
    )


@router.get(
    "/me",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    summary="Current User",
    description="Return the authenticated user's profile.",
)
def me(current_user: dict = Depends(get_current_user)):
    return _user_response(current_user)
