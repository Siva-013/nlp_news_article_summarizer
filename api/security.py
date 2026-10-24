"""
Security primitives for user authentication.

- Passwords are hashed with bcrypt (salted, cost 12); plaintext is never stored.
- Sessions use short-lived signed JWTs (HS256) carried as Bearer tokens.
- The signing secret is read from the JWT_SECRET environment variable; a
  fixed development fallback keeps local sessions valid across restarts but
  MUST be replaced (or overridden via env) for any real deployment.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import bcrypt
import jwt

# bcrypt truncates input at 72 bytes; reject longer passwords up front instead.
MAX_PASSWORD_BYTES = 72

# HS256 keys should be at least 32 bytes (RFC 7518 §3.2).
_DEV_SECRET_FALLBACK = "lexica-dev-secret-change-me-in-production-0123456789abcdef"

TOKEN_ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 24 * 7  # one week


def get_jwt_secret() -> str:
    """Return the JWT signing secret from env, or the documented dev fallback."""
    return os.environ.get("JWT_SECRET") or _DEV_SECRET_FALLBACK


def validate_password_bytes(password: str) -> str:
    """Ensure the password fits bcrypt's 72-byte input limit."""
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise ValueError(
            f"Password is too long (max {MAX_PASSWORD_BYTES} bytes)."
        )
    return password


def hash_password(password: str) -> str:
    """Hash a plaintext password with bcrypt; returns an ascii str."""
    validate_password_bytes(password)
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12))
    return hashed.decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    """Constant-time bcrypt verification; False on any mismatch/malformed hash."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("ascii"))
    except (ValueError, UnicodeError):
        return False


def create_access_token(user_id: int, username: str) -> str:
    """Mint a signed JWT for the given user."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "username": username,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(hours=TOKEN_EXPIRE_HOURS)).timestamp()),
    }
    return jwt.encode(payload, get_jwt_secret(), algorithm=TOKEN_ALGORITHM)


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify signature and expiry; return the payload dict or None when the
    token is invalid, expired, or malformed.
    """
    try:
        payload = jwt.decode(token, get_jwt_secret(), algorithms=[TOKEN_ALGORITHM])
    except jwt.PyJWTError:
        return None
    if not payload or not payload.get("sub"):
        return None
    return payload
