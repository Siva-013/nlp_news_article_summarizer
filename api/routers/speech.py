"""
Text-to-Speech (TTS) Router
POST /api/v1/speech/synthesize
GET  /api/v1/speech/audio/{audio_id}
GET  /api/v1/speech/voices
"""

from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from api.dependencies import (
    get_audio_dir,
    get_current_user,
    get_tts_engine,
    validate_safe_audio_id,
)
from api.schemas import (
    SpeechSynthesizeRequest,
    SpeechSynthesizeResponse,
    SpeechVoicesResponse,
    VoiceInfo,
)
from api.services import run_speech_synthesis
from speech.text_to_speech import TextToSpeechEngine

router = APIRouter(prefix="/speech", tags=["Text-to-Speech"])


@router.post(
    "/synthesize",
    response_model=SpeechSynthesizeResponse,
    status_code=status.HTTP_200_OK,
    summary="Synthesize Text Summary to Speech",
    description="Generate a WAV speech audio file from a summary text using offline pyttsx3 (Phase 10).",
)
def synthesize_speech(
    request: SpeechSynthesizeRequest,
    engine: TextToSpeechEngine = Depends(get_tts_engine),
    audio_dir: Path = Depends(get_audio_dir),
    user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        result = run_speech_synthesis(
            text=request.text,
            audio_dir=audio_dir,
            rate=request.rate or 175,
            volume=request.volume if request.volume is not None else 1.0,
            voice_id=request.voice_id,
            engine=engine,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Speech synthesis error: {exc}",
        ) from exc


@router.get(
    "/audio/{audio_id}",
    response_class=FileResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve Synthesized Audio File",
    description="Stream or download the generated WAV speech audio file.",
)
def get_audio_file(
    audio_id: str,
    audio_dir: Path = Depends(get_audio_dir),
    user: Dict[str, Any] = Depends(get_current_user),
):
    # Validate audio ID to prevent path traversal
    try:
        safe_id = validate_safe_audio_id(audio_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    file_path = (audio_dir / f"{safe_id}.wav").resolve()

    # Enforce strict path containment
    if not str(file_path).startswith(str(audio_dir.resolve())):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")

    if not file_path.is_file() or not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audio file '{audio_id}' not found.",
        )

    return FileResponse(
        path=file_path,
        media_type="audio/wav",
        filename=f"{safe_id}.wav",
    )


@router.get(
    "/voices",
    response_model=SpeechVoicesResponse,
    status_code=status.HTTP_200_OK,
    summary="List Available Speech Synthesis Voices",
    description="List all offline speech synthesis voices supported on the current host system.",
)
def list_available_voices(
    engine: TextToSpeechEngine = Depends(get_tts_engine),
    user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        raw_voices = engine.get_available_voices()
        voices = [
            VoiceInfo(
                id=str(v.get("id", "")),
                name=v.get("name"),
                languages=v.get("languages"),
                gender=v.get("gender"),
            )
            for v in raw_voices
        ]
        return {"total": len(voices), "voices": voices}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query speech synthesis voices: {exc}",
        ) from exc
