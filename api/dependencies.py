"""
FastAPI Dependencies and Singletons
Phase 11: FastAPI Backend Integration
"""

import os
from pathlib import Path
import re
from typing import Any, Dict, Generator, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.security import decode_access_token
from database.notes_history import NotesHistoryManager, DEFAULT_DB_PATH
from speech.text_to_speech import TextToSpeechEngine


# Dedicated managed directory for synthesized speech audio
AUDIO_DIR = Path("static/audio").resolve()
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Audio ID safe regex pattern: alphanumeric, hyphen, underscore only
SAFE_AUDIO_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]+$")

# Singleton TTS engine instance (initialized once, reused across requests)
_tts_engine_singleton: Optional[TextToSpeechEngine] = None


def get_db() -> Generator[NotesHistoryManager, None, None]:
    """
    Dependency provider for NotesHistoryManager.
    Instantiates persistence manager against DEFAULT_DB_PATH.
    Can be easily overridden in tests via app.dependency_overrides[get_db].
    """
    manager = NotesHistoryManager(db_path=DEFAULT_DB_PATH)
    try:
        yield manager
    finally:
        manager.close()


def get_tts_engine() -> TextToSpeechEngine:
    """
    Dependency provider for TextToSpeechEngine.
    Returns a singleton instance to avoid reinitializing pyttsx3 on each request.
    Can be overridden in tests to inject a mock speech engine.
    """
    global _tts_engine_singleton
    if _tts_engine_singleton is None:
        _tts_engine_singleton = TextToSpeechEngine(auto_init=True)
    return _tts_engine_singleton


def get_audio_dir() -> Path:
    """
    Dependency provider for audio storage directory.
    """
    return AUDIO_DIR


def validate_safe_audio_id(audio_id: str) -> str:
    """
    Validate that an audio identifier is strictly alphanumeric/dashes/underscores
    to prevent path traversal attacks.

    Args:
        audio_id: Filename or ID string.

    Returns:
        Cleaned, validated audio_id string.

    Raises:
        ValueError: If audio_id contains invalid characters or directory traversal sequences.
    """
    # Strip potential .wav extension if provided in the URL param
    raw_id = audio_id
    if raw_id.lower().endswith(".wav"):
        raw_id = raw_id[:-4]

    if not raw_id or not SAFE_AUDIO_ID_PATTERN.match(raw_id):
        raise ValueError("Invalid audio ID format. Only alphanumeric characters, hyphens, and underscores are allowed.")

    return raw_id


# Bearer-token extractor for authenticated endpoints (auto-docs the 401 response).
_bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    db: NotesHistoryManager = Depends(get_db),
) -> Dict[str, Any]:
    """
    Resolve the authenticated user from the Authorization: Bearer <jwt> header.

    Returns:
        {"id": int, "username": str, "created_at": str}

    Raises:
        HTTPException 401: Missing, malformed, expired, or invalid token; or
        the referenced user no longer exists.
    """
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Provide a valid Bearer token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if credentials is None or not credentials.credentials:
        raise unauthorized

    payload = decode_access_token(credentials.credentials)
    if not payload:
        raise unauthorized

    try:
        user_id = int(payload["sub"])
    except (TypeError, ValueError):
        raise unauthorized

    user = db.get_user_by_id(user_id)
    if not user:
        raise unauthorized

    return {"id": user["id"], "username": user["username"], "created_at": user["created_at"]}
