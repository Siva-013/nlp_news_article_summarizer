"""
Text-to-Speech (TTS) Package
Part of Phase 10: Text-to-Speech (TTS) Integration

Provides offline speech synthesis for news article summaries.
"""

__all__ = [
    "TextToSpeechEngine",
    "DEFAULT_RATE",
    "DEFAULT_VOLUME",
    "SUPPORTED_EXTENSIONS",
]


def __getattr__(name: str):
    if name in (
        "TextToSpeechEngine",
        "DEFAULT_RATE",
        "DEFAULT_VOLUME",
        "SUPPORTED_EXTENSIONS",
    ):
        import speech.text_to_speech as tts
        return getattr(tts, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
