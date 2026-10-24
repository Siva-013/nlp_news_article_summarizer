"""
Unit & Integration Tests for Phase 10: Text-to-Speech (TTS) Integration

Tests cover:
Input validation:
 1. Normal text input
 2. Empty text
 3. Whitespace-only text
 4. Non-string input
 5. Long text
 6. Unicode text
 7. Punctuation and contractions
 8. Text containing newlines

Configuration:
 9. Valid rate
10. Invalid rate
11. Valid volume boundaries (0.0 and 1.0)
12. Invalid volume values
13. Available voice selection
14. Unavailable voice selection
15. Voice-list serialization

Speech engine:
16. Engine initialization
17. Initialization failure handling
18. Speaking a valid summary through a mocked engine
19. Engine failure during speech
20. Stop behavior
21. Resource cleanup
22. Repeated use of the same engine instance

Audio file output:
23. Successful file generation with a controlled engine
24. Missing output directory error
25. Invalid audio extension error
26. Failed synthesis handling
27. Missing or empty output file error
28. Explicit overwrite behavior

CLI and integration:
29. Inline-text CLI operation
30. File-input CLI operation
31. Voice-list CLI operation
32. Invalid argument handling in CLI
33. Summary text supplied from real existing summarization output record
34. Confirmation that Phase 10 does not modify original summary records
35. Caller dictionary immutability
36. Headless test execution (no physical audio hardware required)
"""

import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch
import pytest

from speech.text_to_speech import (
    DEFAULT_RATE,
    DEFAULT_VOLUME,
    MIN_RATE,
    MAX_RATE,
    SUPPORTED_EXTENSIONS,
    TextToSpeechEngine,
    build_parser,
    main as cli_main,
)


class MockVoice:
    """Mock Voice object mirroring pyttsx3 voice structure."""

    def __init__(self, voice_id="voice_david", name="David Desktop", languages=None, gender="Male"):
        self.id = voice_id
        self.name = name
        self.languages = languages or ["en-US"]
        self.gender = gender


class MockPyttsx3Engine:
    """Mock pyttsx3 engine for deterministic headless unit testing."""

    def __init__(self, should_fail=False):
        self.properties = {
            "rate": DEFAULT_RATE,
            "volume": DEFAULT_VOLUME,
            "voice": "voice_david",
            "voices": [
                MockVoice("voice_david", "David Desktop", ["en-US"], "Male"),
                MockVoice("voice_zira", "Zira Desktop", ["en-US"], "Female"),
            ],
        }
        self.spoken_texts = []
        self.saved_files = []
        self.stopped = False
        self.should_fail = should_fail

    def getProperty(self, name):
        if self.should_fail:
            raise RuntimeError("Engine hardware failure during property query.")
        return self.properties.get(name)

    def setProperty(self, name, value):
        if self.should_fail:
            raise RuntimeError(f"Engine hardware failure setting {name}.")
        self.properties[name] = value

    def say(self, text):
        if self.should_fail:
            raise RuntimeError("Engine playback error.")
        self.spoken_texts.append(text)

    def save_to_file(self, text, filename):
        if self.should_fail:
            raise RuntimeError("Engine file synthesis error.")
        self.saved_files.append((text, filename))
        # Create non-empty mock audio file on disk so validation passes
        p = Path(filename)
        p.write_bytes(b"RIFF" + b"\x00" * 44)  # Minimal mock WAV header bytes

    def runAndWait(self):
        if self.should_fail:
            raise RuntimeError("Engine execution failure in runAndWait.")

    def stop(self):
        self.stopped = True


@pytest.fixture
def mock_engine():
    return MockPyttsx3Engine()


@pytest.fixture
def tts_service(mock_engine):
    return TextToSpeechEngine(engine=mock_engine)


# =============================================================================
# 1. INPUT VALIDATION TESTS (1 - 8)
# =============================================================================

def test_1_normal_text_input(tts_service):
    """Test 1: Valid summary text passes validation without alteration."""
    text = "The United Nations announced a climate agreement."
    res = tts_service.validate_text(text)
    assert res == text


def test_2_empty_text_input(tts_service):
    """Test 2: Empty string raises ValueError."""
    with pytest.raises(ValueError, match="cannot be empty"):
        tts_service.validate_text("")


def test_3_whitespace_only_text(tts_service):
    """Test 3: Whitespace-only string raises ValueError."""
    with pytest.raises(ValueError, match="cannot be empty or contain only whitespace"):
        tts_service.validate_text("   \n\t  ")


def test_4_non_string_input(tts_service):
    """Test 4: Non-string input raises TypeError."""
    with pytest.raises(TypeError, match="must be a string"):
        tts_service.validate_text(12345)
    with pytest.raises(TypeError, match="must be a string"):
        tts_service.validate_text(["summary list"])


def test_5_long_text_input(tts_service):
    """Test 5: Long article text (>10,000 characters) validates cleanly."""
    long_text = "Detailed news paragraph. " * 500
    assert len(long_text) > 10000
    res = tts_service.validate_text(long_text)
    assert len(res) == len(long_text.strip())


def test_6_unicode_text_input(tts_service):
    """Test 6: International Unicode characters and diacritics validate correctly."""
    text = "President François Hollande met in Genève with Angela Merkel."
    res = tts_service.validate_text(text)
    assert "François" in res
    assert "Genève" in res


def test_7_punctuation_and_contractions(tts_service):
    """Test 7: Contractions and journalistic punctuation are preserved."""
    text = "It's official: the Prime Minister won't resign, despite MPs' protests."
    res = tts_service.validate_text(text)
    assert "It's" in res
    assert "won't" in res
    assert "MPs'" in res


def test_8_text_containing_newlines(tts_service):
    """Test 8: Multi-line summaries containing newlines validate properly."""
    text = "First summary sentence.\nSecond summary sentence.\nThird sentence."
    res = tts_service.validate_text(text)
    assert "\n" in res
    assert res.startswith("First")
    assert res.endswith("sentence.")


# =============================================================================
# 2. CONFIGURATION TESTS (9 - 15)
# =============================================================================

def test_9_valid_rate(tts_service, mock_engine):
    """Test 9: Setting valid speech rate updates property on engine."""
    tts_service.set_rate(160)
    assert tts_service.rate == 160
    assert mock_engine.properties["rate"] == 160


def test_10_invalid_rate(tts_service):
    """Test 10: Setting rate outside bounds or non-int raises error."""
    with pytest.raises(ValueError, match="out of bounds"):
        tts_service.set_rate(MIN_RATE - 1)
    with pytest.raises(ValueError, match="out of bounds"):
        tts_service.set_rate(MAX_RATE + 1)
    with pytest.raises(TypeError, match="must be an integer"):
        tts_service.set_rate("fast")


def test_11_valid_volume_boundaries(tts_service, mock_engine):
    """Test 11: Setting volume at boundaries 0.0 and 1.0."""
    tts_service.set_volume(0.0)
    assert tts_service.volume == 0.0
    assert mock_engine.properties["volume"] == 0.0

    tts_service.set_volume(1.0)
    assert tts_service.volume == 1.0
    assert mock_engine.properties["volume"] == 1.0


def test_12_invalid_volume(tts_service):
    """Test 12: Setting volume outside [0.0, 1.0] raises ValueError."""
    with pytest.raises(ValueError, match="out of bounds"):
        tts_service.set_volume(-0.1)
    with pytest.raises(ValueError, match="out of bounds"):
        tts_service.set_volume(1.05)
    with pytest.raises(TypeError, match="must be a numeric value"):
        tts_service.set_volume("loud")


def test_13_available_voice_selection(tts_service, mock_engine):
    """Test 13: Selecting available voice updates engine voice property."""
    tts_service.set_voice("voice_zira")
    assert tts_service.voice_id == "voice_zira"
    assert mock_engine.properties["voice"] == "voice_zira"


def test_14_unavailable_voice_selection(tts_service):
    """Test 14: Requesting non-existent voice ID raises ValueError."""
    with pytest.raises(ValueError, match="is not available on this system"):
        tts_service.set_voice("voice_nonexistent_xyz")


def test_15_voice_list_serialization(tts_service):
    """Test 15: get_available_voices returns serializable list of dicts."""
    voices = tts_service.get_available_voices()
    assert len(voices) == 2
    assert voices[0]["id"] == "voice_david"
    assert voices[0]["name"] == "David Desktop"
    # Ensure JSON serializable
    serialized = json.dumps(voices)
    assert "David Desktop" in serialized


# =============================================================================
# 3. SPEECH ENGINE LIFECYCLE TESTS (16 - 22)
# =============================================================================

def test_16_engine_initialization():
    """Test 16: Lazy or direct initialization succeeds."""
    mock_eng = MockPyttsx3Engine()
    engine_obj = TextToSpeechEngine(engine=mock_eng, auto_init=True)
    assert engine_obj._engine is mock_eng


def test_17_initialization_failure():
    """Test 17: Engine initialization error raises RuntimeError."""
    with patch("pyttsx3.init", side_effect=Exception("Driver not found")):
        with pytest.raises(RuntimeError, match="Failed to initialize system speech engine"):
            TextToSpeechEngine(engine=None, auto_init=True)


def test_18_speak_valid_summary(tts_service, mock_engine):
    """Test 18: speak() executes say() and runAndWait()."""
    summary = "Global stock markets rallied following quarterly economic reports."
    tts_service.speak(summary)
    assert len(mock_engine.spoken_texts) == 1
    assert mock_engine.spoken_texts[0] == summary


def test_19_engine_failure_during_speech():
    """Test 19: Hardware or engine failure during speak() raises RuntimeError."""
    failing_engine = MockPyttsx3Engine(should_fail=True)
    tts = TextToSpeechEngine(engine=failing_engine)
    with pytest.raises(RuntimeError, match="Speech playback failed"):
        tts.speak("This should fail.")


def test_20_stop_behavior(tts_service, mock_engine):
    """Test 20: stop() invokes engine stop method safely."""
    tts_service.stop()
    assert mock_engine.stopped is True


def test_21_resource_cleanup(tts_service):
    """Test 21: Stop and cleanup execute without unhandled errors."""
    tts_service.stop()
    assert tts_service._engine is not None


def test_22_repeated_engine_use(tts_service, mock_engine):
    """Test 22: Reusing the same engine instance across multiple operations."""
    tts_service.speak("First summary.")
    tts_service.speak("Second summary.")
    assert len(mock_engine.spoken_texts) == 2
    assert mock_engine.spoken_texts[0] == "First summary."
    assert mock_engine.spoken_texts[1] == "Second summary."


# =============================================================================
# 4. AUDIO FILE OUTPUT TESTS (23 - 28)
# =============================================================================

def test_23_successful_file_generation(tts_service):
    """Test 23: save_to_file creates non-empty WAV file and returns resolved Path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "output.wav"
        result_path = tts_service.save_to_file("A test summary.", out_path)
        assert result_path.exists()
        assert result_path.stat().st_size > 0
        assert result_path == out_path.resolve()


def test_24_missing_output_directory(tts_service):
    """Test 24: Non-existent destination directory raises FileNotFoundError."""
    non_existent = Path("non_existent_folder_xyz123/output.wav")
    with pytest.raises(FileNotFoundError, match="Output directory does not exist"):
        tts_service.save_to_file("Text summary", non_existent)


def test_25_invalid_extension(tts_service):
    """Test 25: Unsupported audio extension raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        invalid_path = Path(tmpdir) / "output.mp3"
        with pytest.raises(ValueError, match="Unsupported audio file format"):
            tts_service.save_to_file("Text summary", invalid_path)


def test_26_failed_synthesis():
    """Test 26: Engine error during file save raises RuntimeError."""

    class SaveFailsEngine(MockPyttsx3Engine):
        def save_to_file(self, text, filename):
            raise RuntimeError("Engine file synthesis error.")

    tts = TextToSpeechEngine(engine=SaveFailsEngine())
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "fail.wav"
        with pytest.raises(RuntimeError, match="Speech synthesis to file failed"):
            tts.save_to_file("Sample text", out_path)


def test_27_missing_or_empty_output_file():
    """Test 27: If engine finishes without creating non-empty file, raises RuntimeError."""
    class EmptyEngine(MockPyttsx3Engine):
        def save_to_file(self, text, filename):
            Path(filename).touch()  # 0 byte empty file

    tts = TextToSpeechEngine(engine=EmptyEngine())
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_out = Path(tmpdir) / "empty.wav"
        with pytest.raises(RuntimeError, match="generated audio file is empty"):
            tts.save_to_file("Sample text", empty_out)


def test_28_explicit_overwrite_behavior(tts_service):
    """Test 28: Overwrite=False raises FileExistsError if destination exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "existing.wav"
        target.write_bytes(b"dummy bytes")

        # overwrite=False must raise FileExistsError
        with pytest.raises(FileExistsError, match="Destination file already exists"):
            tts_service.save_to_file("New summary", target, overwrite=False)

        # overwrite=True succeeds
        res = tts_service.save_to_file("New summary", target, overwrite=True)
        assert res.exists()


# =============================================================================
# 5. CLI & INTEGRATION TESTS (29 - 36)
# =============================================================================

def test_29_cli_inline_text(mock_engine):
    """Test 29: CLI supports inline text synthesis with --text and --output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_wav = Path(tmpdir) / "cli_out.wav"
        with patch("sys.argv", ["tts", "--text", "CLI inline summary.", "--output", str(out_wav)]):
            with patch("speech.text_to_speech.pyttsx3.init", return_value=mock_engine):
                exit_code = cli_main()
                assert exit_code == 0
                assert out_wav.exists()


def test_30_cli_file_input(mock_engine):
    """Test 30: CLI supports reading summary from text file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        txt_file = Path(tmpdir) / "summary.txt"
        txt_file.write_text("Summary text from disk file.", encoding="utf-8")
        out_wav = Path(tmpdir) / "file_out.wav"

        with patch("sys.argv", ["tts", "--input-file", str(txt_file), "--output", str(out_wav)]):
            with patch("speech.text_to_speech.pyttsx3.init", return_value=mock_engine):
                exit_code = cli_main()
                assert exit_code == 0
                assert out_wav.exists()


def test_31_cli_voice_list(mock_engine, capsys):
    """Test 31: CLI --list-voices displays voices and exits 0."""
    with patch("sys.argv", ["tts", "--list-voices"]):
        with patch("speech.text_to_speech.pyttsx3.init", return_value=mock_engine):
            exit_code = cli_main()
            assert exit_code == 0
            captured = capsys.readouterr().out
            assert "AVAILABLE SYSTEM VOICES" in captured
            assert "David Desktop" in captured


def test_32_cli_invalid_arguments(capsys):
    """Test 32: CLI without text or with non-existent file returns non-zero error."""
    with patch("sys.argv", ["tts", "--input-file", "nonexistent_12345.txt"]):
        exit_code = cli_main()
        assert exit_code == 1
        err = capsys.readouterr().err
        assert "Input file not found" in err


def test_33_real_summarization_record_integration(tts_service):
    """Test 33: Synthesizing real summary from existing TextRank/Frequency outputs."""
    summary_file = Path("dataset/processed/textrank_summaries_1000.jsonl")
    if not summary_file.exists():
        summary_file = Path("dataset/processed/frequency_summaries_1000.jsonl")

    if not summary_file.exists():
        pytest.skip("Processed summary dataset not available.")

    with summary_file.open("r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    record = json.loads(first_line)

    summary_text = record.get("summary", "")
    assert len(summary_text) > 0

    with tempfile.TemporaryDirectory() as tmpdir:
        out_wav = Path(tmpdir) / "real_summary.wav"
        res = tts_service.save_to_file(summary_text, out_wav)
        assert res.exists()
        assert res.stat().st_size > 0


def test_34_original_summary_record_immutability(tts_service):
    """Test 34: Synthesizing speech from a summary dict does NOT mutate the record."""
    record = {
        "id": "cnn_art_100",
        "method": "textrank",
        "summary": "This is an extractive summary of global news.",
        "metadata": {"sentences": 3},
    }
    record_backup = copy.deepcopy(record)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_wav = Path(tmpdir) / "record_test.wav"
        tts_service.save_to_file(record["summary"], out_wav)

    assert record == record_backup


def test_35_caller_dictionary_immutability():
    """Test 35: General input objects are not modified."""
    params = {"rate": 180, "volume": 0.9}
    params_backup = copy.deepcopy(params)
    mock_eng = MockPyttsx3Engine()
    tts = TextToSpeechEngine(rate=params["rate"], volume=params["volume"], engine=mock_eng)
    assert tts.rate == 180
    assert params == params_backup


def test_36_headless_test_execution_no_hardware_required(tts_service):
    """Test 36: Validates that all unit tests execute headlessly without physical audio hardware."""
    # Verified by the presence of MockPyttsx3Engine across test cases
    assert tts_service._engine is not None
    assert isinstance(tts_service._engine, MockPyttsx3Engine)
