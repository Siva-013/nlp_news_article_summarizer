"""
Integration test for real pyttsx3 speech synthesis on Windows.
This test uses the actual system TTS engine (not mocked) to verify Windows SAPI5 works.
"""

import pytest
from pathlib import Path
import tempfile
from speech.text_to_speech import TextToSpeechEngine


def test_real_tts_engine_initialization():
    """Test that real pyttsx3 engine initializes successfully on Windows."""
    engine = TextToSpeechEngine(auto_init=True)
    assert engine._engine is not None
    voices = engine.get_available_voices()
    assert len(voices) > 0
    print(f"Found {len(voices)} voices: {[v['name'] for v in voices]}")


def test_real_tts_synthesis():
    """Test real speech synthesis to WAV file."""
    engine = TextToSpeechEngine(auto_init=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_speech.wav"
        result = engine.save_to_file(
            "This is a text to speech test.",
            output_path,
            overwrite=True
        )
        assert result.exists()
        assert result.stat().st_size > 0
        print(f"Generated WAV file: {result.stat().st_size} bytes")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
