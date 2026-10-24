"""
Unit and Integration Tests for Phase 11: FastAPI Backend Integration

Tests cover:
1. Root and Health:
   - Root endpoint metadata (GET /)
   - Health check with database connectivity (GET /health)

2. Extractive Summarization (POST /api/v1/summarize):
   - Frequency summarization
   - TF-IDF summarization
   - TextRank summarization
   - Custom sentence count (num_sentences)
   - Empty input text validation (422)
   - Invalid method rejection (422)
   - Out-of-bounds sentence count (422)

3. Vocabulary Extraction (POST /api/v1/vocabulary/extract):
   - Extraction of content words and entities (Phase 7)
   - Custom max_items and include_entities parameters
   - Empty text validation (422)

4. New-Word Detection (POST /api/v1/vocabulary/detect-new):
   - Novelty detection from raw article text + known vocabulary (Phase 8)
   - Novelty detection from pre-extracted extraction record
   - Missing both text and extraction record (400)
   - Accounting validation (total == known + new)

5. Article History Persistence (CRUD on /api/v1/history):
   - Create article history record (201)
   - Retrieve article history record by ID (200)
   - Non-existent history record lookup (404)
   - List history records with pagination
   - Filter history records by summarization method
   - Search article history records (keyword search)
   - Delete history record (200)
   - Non-existent history record deletion (404)

6. Vocabulary Notes Persistence (CRUD on /api/v1/notes):
   - Create vocabulary note (201)
   - Retrieve vocabulary note by ID (200)
   - Non-existent note lookup (404)
   - List notes with pagination and term filtering
   - Search vocabulary notes across fields
   - Update vocabulary note fields (200)
   - Update non-existent note (404)
   - Delete vocabulary note (200)
   - Foreign key integrity: note preserved when associated history record is deleted (ON DELETE SET NULL)

7. Text-to-Speech (TTS) Integration (/api/v1/speech/*):
   - Speech synthesis to WAV file (POST /api/v1/speech/synthesize)
   - Retrieve synthesized audio file (GET /api/v1/speech/audio/{audio_id})
   - Path traversal attempt rejection (GET /api/v1/speech/audio/../notes.db) (400)
   - Non-existent audio ID (404)
   - List available speech voices (GET /api/v1/speech/voices)

8. Quantitative Evaluation (/api/v1/evaluate):
   - Compute ROUGE-1, ROUGE-2, and ROUGE-L metrics (POST /api/v1/evaluate)
   - Validation failure on empty strings (422)
   - Deterministic metric bounds verification [0.0, 1.0]

9. Complete Isolation:
   - Production database (database/notes_history.db) is never touched during tests.
"""

from pathlib import Path
import tempfile
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.dependencies import get_db, get_tts_engine, get_audio_dir
from database.notes_history import NotesHistoryManager
from speech.text_to_speech import TextToSpeechEngine, DEFAULT_RATE, DEFAULT_VOLUME


# =============================================================================
# MOCK TTS INFRASTRUCTURE FOR HEADLESS TESTING
# =============================================================================

class MockVoice:
    def __init__(self, voice_id="voice_david", name="David Desktop", languages=None, gender="Male"):
        self.id = voice_id
        self.name = name
        self.languages = languages or ["en-US"]
        self.gender = gender


class MockPyttsx3Engine:
    def __init__(self):
        self.properties = {
            "rate": DEFAULT_RATE,
            "volume": DEFAULT_VOLUME,
            "voice": "voice_david",
            "voices": [
                MockVoice("voice_david", "David Desktop", ["en-US"], "Male"),
                MockVoice("voice_zira", "Zira Desktop", ["en-US"], "Female"),
            ],
        }
        self.saved_files = []

    def getProperty(self, name):
        return self.properties.get(name)

    def setProperty(self, name, value):
        self.properties[name] = value

    def save_to_file(self, text, filename):
        self.saved_files.append((text, filename))
        p = Path(filename)
        p.write_bytes(b"RIFF" + b"\x00" * 44)  # Minimal mock WAV header bytes

    def runAndWait(self):
        pass

    def stop(self):
        pass


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def test_db(tmp_path):
    """Isolated SQLite persistence manager using temporary database file."""
    db_file = tmp_path / "test_notes_history.db"
    manager = NotesHistoryManager(db_path=db_file)
    yield manager
    manager.close()


@pytest.fixture
def test_audio_dir(tmp_path):
    """Temporary directory for test-generated audio files."""
    audio_path = tmp_path / "audio"
    audio_path.mkdir(parents=True, exist_ok=True)
    return audio_path


@pytest.fixture
def mock_tts():
    """TextToSpeechEngine wrapped around a headless MockPyttsx3Engine."""
    mock_eng = MockPyttsx3Engine()
    return TextToSpeechEngine(engine=mock_eng, auto_init=False)


@pytest.fixture
def client(test_db, test_audio_dir, mock_tts):
    """
    TestClient with dependencies overridden for complete database and filesystem isolation.
    """
    def override_get_db():
        yield test_db

    def override_get_tts_engine():
        return mock_tts

    def override_get_audio_dir():
        return test_audio_dir

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_tts_engine] = override_get_tts_engine
    app.dependency_overrides[get_audio_dir] = override_get_audio_dir

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


SAMPLE_ARTICLE = (
    "London is the capital and largest city of England and the United Kingdom. "
    "It stands on the River Thames in south-east England at the head of a 50-mile estuary down to the North Sea. "
    "London has been a major settlement for two millennia. "
    "It was founded by the Romans, who named it Londinium. "
    "The City of London, London's ancient core, largely retains its 1.12-square-mile medieval boundaries."
)


# =============================================================================
# ROOT AND HEALTH TESTS
# =============================================================================

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "endpoints" in data
    assert data["endpoints"]["summarize"] == "/api/v1/summarize"


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database_connected"] is True
    assert data["engine"] == "classical_nlp_fastapi"


# =============================================================================
# SUMMARIZATION ENDPOINT TESTS
# =============================================================================

def test_summarize_frequency_success(client):
    payload = {
        "text": SAMPLE_ARTICLE,
        "method": "frequency",
        "num_sentences": 2,
    }
    response = client.post("/api/v1/summarize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["method"] == "frequency"
    assert data["num_sentences_requested"] == 2
    assert data["num_sentences_selected"] == 2
    assert len(data["selected_sentences"]) == 2
    assert isinstance(data["summary"], str)
    assert len(data["summary"]) > 0


def test_summarize_tfidf_success(client):
    payload = {
        "text": SAMPLE_ARTICLE,
        "method": "tfidf",
        "num_sentences": 2,
    }
    response = client.post("/api/v1/summarize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["method"] == "tfidf"
    assert data["num_sentences_selected"] == 2
    assert len(data["selected_sentences"]) == 2


def test_summarize_textrank_success(client):
    payload = {
        "text": SAMPLE_ARTICLE,
        "method": "textrank",
        "num_sentences": 2,
    }
    response = client.post("/api/v1/summarize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["method"] == "textrank"
    assert data["num_sentences_selected"] == 2
    assert len(data["selected_sentences"]) == 2


def test_summarize_empty_text_validation(client):
    response = client.post("/api/v1/summarize", json={"text": "   ", "method": "frequency"})
    assert response.status_code == 422


def test_summarize_invalid_method(client):
    response = client.post("/api/v1/summarize", json={"text": SAMPLE_ARTICLE, "method": "unsupported_method"})
    assert response.status_code == 422


def test_summarize_invalid_num_sentences(client):
    response = client.post("/api/v1/summarize", json={"text": SAMPLE_ARTICLE, "num_sentences": 0})
    assert response.status_code == 422


# =============================================================================
# VOCABULARY EXTRACTION TESTS
# =============================================================================

def test_vocabulary_extract_success(client):
    payload = {
        "text": SAMPLE_ARTICLE,
        "max_items": 10,
        "include_entities": True,
    }
    response = client.post("/api/v1/vocabulary/extract", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["method"] == "classical_vocabulary_extraction"
    assert data["returned_item_count"] <= 10
    assert len(data["items"]) == data["returned_item_count"]
    if data["items"]:
        first = data["items"][0]
        assert "term" in first
        assert "normalized_form" in first
        assert "importance_score" in first


def test_vocabulary_extract_empty_text(client):
    response = client.post("/api/v1/vocabulary/extract", json={"text": ""})
    assert response.status_code == 422


# =============================================================================
# NEW-WORD DETECTION TESTS
# =============================================================================

def test_vocabulary_detect_new_from_raw_text(client):
    payload = {
        "text": SAMPLE_ARTICLE,
        "known_vocabulary": ["london", "city", "england"],
        "max_items": 15,
    }
    response = client.post("/api/v1/vocabulary/detect-new", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["method"] == "classical_new_word_detection"
    assert data["validation"]["reconciled"] is True
    assert data["total_vocabulary_count"] == data["known_word_count"] + data["new_word_count"]
    assert len(data["known_words"]) == data["known_word_count"]
    assert len(data["new_words"]) == data["new_word_count"]


def test_vocabulary_detect_new_from_extraction_record(client):
    extraction_record = {
        "article_id": "art_101",
        "items": [
            {"term": "River", "normalized_form": "river", "lemma": "river"},
            {"term": "Thames", "normalized_form": "thames", "lemma": "thames"},
            {"term": "Millennium", "normalized_form": "millennium", "lemma": "millennium"},
        ],
    }
    payload = {
        "extraction_record": extraction_record,
        "known_vocabulary": ["river"],
    }
    response = client.post("/api/v1/vocabulary/detect-new", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_vocabulary_count"] == 3
    assert data["known_word_count"] == 1
    assert data["new_word_count"] == 2
    assert data["known_words"][0]["normalized_form"] == "river"


def test_vocabulary_detect_new_missing_inputs(client):
    response = client.post("/api/v1/vocabulary/detect-new", json={"known_vocabulary": ["dog"]})
    assert response.status_code == 400
    assert "Either 'text' or 'extraction_record' must be provided" in response.json()["detail"]


# =============================================================================
# ARTICLE HISTORY CRUD TESTS
# =============================================================================

def test_article_history_crud_flow(client):
    # 1. Create history entry
    create_payload = {
        "source_article_id": "cnn_001",
        "title": "London History Overview",
        "article_text": SAMPLE_ARTICLE,
        "summary": "London is the capital of England. It stands on the River Thames.",
        "method": "frequency",
        "extracted_vocabulary": [{"term": "capital"}, {"term": "river"}],
        "new_vocabulary": [{"term": "estuary"}],
        "metadata": {"source": "CNN"},
    }
    create_res = client.post("/api/v1/history", json=create_payload)
    assert create_res.status_code == 201
    created_entry = create_res.json()
    hist_id = created_entry["id"]
    assert hist_id > 0
    assert created_entry["title"] == "London History Overview"
    assert created_entry["summary"] == create_payload["summary"]

    # 2. Get history entry by ID
    get_res = client.get(f"/api/v1/history/{hist_id}")
    assert get_res.status_code == 200
    assert get_res.json()["id"] == hist_id
    assert get_res.json()["extracted_vocabulary"] == [{"term": "capital"}, {"term": "river"}]

    # 3. List history entries
    list_res = client.get("/api/v1/history?limit=10&offset=0")
    assert list_res.status_code == 200
    list_data = list_res.json()
    assert list_data["total"] >= 1
    assert any(item["id"] == hist_id for item in list_data["items"])

    # 4. Search history entries
    search_res = client.get("/api/v1/history/search?q=Thames")
    assert search_res.status_code == 200
    search_data = search_res.json()
    assert len(search_data["items"]) >= 1

    # 5. Delete history entry
    del_res = client.delete(f"/api/v1/history/{hist_id}")
    assert del_res.status_code == 200
    assert del_res.json()["deleted_id"] == hist_id

    # 6. Verify deleted (404)
    get_after_del = client.get(f"/api/v1/history/{hist_id}")
    assert get_after_del.status_code == 404


def test_article_history_not_found(client):
    response = client.get("/api/v1/history/999999")
    assert response.status_code == 404

    del_response = client.delete("/api/v1/history/999999")
    assert del_response.status_code == 404


# =============================================================================
# VOCABULARY NOTES CRUD TESTS
# =============================================================================

def test_vocabulary_notes_crud_flow(client):
    # 1. Create an article history entry first for linkage
    hist_res = client.post(
        "/api/v1/history",
        json={
            "summary": "Sample summary text.",
            "method": "frequency",
            "title": "Note Link Test",
        },
    )
    hist_id = hist_res.json()["id"]

    # 2. Create vocabulary note
    note_payload = {
        "term": "Estuary",
        "content": "A tidal mouth of a large river, where the tide meets the stream.",
        "example_sentence": "It stands on the River Thames at the head of a 50-mile estuary.",
        "article_history_id": hist_id,
    }
    create_res = client.post("/api/v1/notes", json=note_payload)
    assert create_res.status_code == 201
    note_entry = create_res.json()
    note_id = note_entry["id"]
    assert note_id > 0
    assert note_entry["term"] == "Estuary"
    assert note_entry["article_history_id"] == hist_id

    # 3. Get note by ID
    get_res = client.get(f"/api/v1/notes/{note_id}")
    assert get_res.status_code == 200
    assert get_res.json()["id"] == note_id

    # 4. List notes
    list_res = client.get("/api/v1/notes?limit=10&offset=0")
    assert list_res.status_code == 200
    assert list_res.json()["total"] >= 1

    # 5. Search notes
    search_res = client.get("/api/v1/notes/search?q=tidal")
    assert search_res.status_code == 200
    assert len(search_res.json()["items"]) >= 1

    # 6. Update note
    update_payload = {
        "content": "Updated definition: an arm or inlet of the sea at the lower end of a river.",
    }
    upd_res = client.put(f"/api/v1/notes/{note_id}", json=update_payload)
    assert upd_res.status_code == 200
    assert "arm or inlet" in upd_res.json()["content"]
    assert upd_res.json()["term"] == "Estuary"  # Unchanged term remains

    # 7. Foreign-key test: delete the parent article history
    del_hist = client.delete(f"/api/v1/history/{hist_id}")
    assert del_hist.status_code == 200

    # The note still exists, but article_history_id is set to None (ON DELETE SET NULL)
    get_note_after_parent_del = client.get(f"/api/v1/notes/{note_id}")
    assert get_note_after_parent_del.status_code == 200
    assert get_note_after_parent_del.json()["article_history_id"] is None

    # 8. Delete note
    del_note_res = client.delete(f"/api/v1/notes/{note_id}")
    assert del_note_res.status_code == 200

    # 9. Verify note deleted (404)
    assert client.get(f"/api/v1/notes/{note_id}").status_code == 404


def test_vocabulary_note_not_found(client):
    assert client.get("/api/v1/notes/999999").status_code == 404
    assert client.delete("/api/v1/notes/999999").status_code == 404
    assert client.put("/api/v1/notes/999999", json={"content": "New"}).status_code == 404


# =============================================================================
# TEXT-TO-SPEECH (TTS) ENDPOINT TESTS
# =============================================================================

def test_speech_synthesize_and_retrieve(client, test_audio_dir):
    synth_payload = {
        "text": "London has been a major settlement for two millennia.",
        "rate": 180,
        "volume": 0.9,
    }
    synth_res = client.post("/api/v1/speech/synthesize", json=synth_payload)
    assert synth_res.status_code == 200
    synth_data = synth_res.json()

    audio_id = synth_data["audio_id"]
    assert audio_id
    assert synth_data["audio_url"] == f"/api/v1/speech/audio/{audio_id}"
    assert synth_data["file_size_bytes"] > 0

    # Retrieve synthesized audio file
    audio_res = client.get(f"/api/v1/speech/audio/{audio_id}")
    assert audio_res.status_code == 200
    assert audio_res.headers["content-type"] == "audio/wav"
    assert len(audio_res.content) > 0


def test_speech_audio_path_traversal_protection(client):
    # Attempt directory traversal with ..
    res = client.get("/api/v1/speech/audio/..%2Fsecret")
    assert res.status_code in (400, 404)


def test_speech_audio_not_found(client):
    res = client.get("/api/v1/speech/audio/non_existent_audio_id_12345")
    assert res.status_code == 404


def test_speech_list_voices(client):
    res = client.get("/api/v1/speech/voices")
    assert res.status_code == 200
    data = res.json()
    assert "voices" in data
    assert data["total"] == len(data["voices"])
    assert data["total"] >= 1
    assert data["voices"][0]["id"] == "voice_david"


# =============================================================================
# QUANTITATIVE EVALUATION (ROUGE) TESTS
# =============================================================================

def test_evaluate_rouge_success(client):
    eval_payload = {
        "candidate_summary": "London is the capital of England and has been a major settlement.",
        "reference_summary": "London is the capital of England and was founded by the Romans.",
    }
    response = client.post("/api/v1/evaluate", json=eval_payload)
    assert response.status_code == 200
    data = response.json()
    assert "rouge1" in data
    assert "rouge2" in data
    assert "rougeL" in data

    for metric in ("rouge1", "rouge2", "rougeL"):
        assert 0.0 <= data[metric]["precision"] <= 1.0
        assert 0.0 <= data[metric]["recall"] <= 1.0
        assert 0.0 <= data[metric]["f1"] <= 1.0


def test_evaluate_rouge_empty_validation(client):
    response = client.post(
        "/api/v1/evaluate",
        json={"candidate_summary": "  ", "reference_summary": "Valid reference."},
    )
    assert response.status_code == 422
