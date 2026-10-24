"""
Integration test for real TTS API endpoints without mocking.
Tests the complete flow from API request to WAV file generation using real pyttsx3.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app


def test_real_tts_voices_endpoint():
    """Test the voices endpoint with real pyttsx3 engine."""
    with TestClient(app) as client:
        response = client.get("/api/v1/speech/voices")
        assert response.status_code == 200
        data = response.json()
        assert "voices" in data
        assert data["total"] == len(data["voices"])
        assert data["total"] > 0
        print(f"Voices endpoint returned {data['total']} voices")


def test_real_tts_synthesize_endpoint():
    """Test the synthesize endpoint with real pyttsx3 engine."""
    with TestClient(app) as client:
        payload = {
            "text": "This is a text to speech test.",
            "rate": 175,
            "volume": 1.0,
        }
        response = client.post("/api/v1/speech/synthesize", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "audio_id" in data
        assert "audio_url" in data
        assert data["file_size_bytes"] > 0
        print(f"Synthesized audio: {data['file_name']} ({data['file_size_bytes']} bytes)")

        # Retrieve the audio file
        audio_response = client.get(data["audio_url"])
        assert audio_response.status_code == 200
        assert audio_response.headers["content-type"] == "audio/wav"
        assert len(audio_response.content) > 0
        print(f"Retrieved audio file: {len(audio_response.content)} bytes")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
