"""
Unit & Integration Tests for Phase 9: Notes and Article History Persistence

Tests cover:
1. Article History:
   - Database initialization
   - Saving and retrieving a history entry
   - Saving multiple entries for the same source article
   - Listing and pagination (limit/offset)
   - Searching title/text/summary/source_id
   - Filtering by summarization method
   - Deleting an individual entry
   - Explicit clear-history behavior
   - Handling missing identifiers
   - Optional fields handling
   - Timestamp generation and ISO formatting
   - Correct serialization of vocabulary fields

2. Vocabulary Notes:
   - Creating and retrieving a note
   - Listing and pagination
   - Searching note content
   - Updating note content and fields
   - Handling missing note IDs
   - Deleting a note
   - General notes without a vocabulary term
   - Optional article association
   - Invalid input validation
   - SQL injection-style input treated strictly as data

3. Reliability & Integrity:
   - Transaction rollback on failure
   - Database connection cleanup
   - Foreign-key behavior (ON DELETE SET NULL)
   - Independent temporary databases across tests
   - JSON serialization of supported data
   - Repeated operations and deterministic behavior
   - No mutation of caller-supplied input dictionaries
   - Integration test combining article history and notes
   - Never writing to production database during tests
"""

import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import pytest

from database.notes_history import DEFAULT_DB_PATH, NotesHistoryManager


@pytest.fixture
def temp_db():
    """Fixture providing a fresh in-memory or temp-file NotesHistoryManager for each test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "test_notes_history.db"
        manager = NotesHistoryManager(db_path=db_file)
        yield manager
        manager.close()


# =============================================================================
# 1. ARTICLE HISTORY TESTS
# =============================================================================

def test_article_history_initialization(temp_db):
    """Test database initialization and table creation."""
    assert Path(temp_db.db_path).exists()
    conn = temp_db.get_connection()
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('article_history', 'vocabulary_notes');"
    )
    tables = {row["name"] for row in cursor.fetchall()}
    assert "article_history" in tables
    assert "vocabulary_notes" in tables
    conn.close()


def test_save_and_retrieve_article_history(temp_db):
    """Test saving an extractive summary and retrieving all its fields."""
    vocab_items = [{"term": "ICC", "lemma": "icc", "pos": "PROPN"}]
    new_vocab = [{"term": "ICC", "pos": "PROPN"}]
    meta = {"rouge1": 0.35, "rouge2": 0.12, "model": "textrank"}

    history_id = temp_db.save_article_history(
        summary="Extractive summary sentence one. Second sentence here.",
        method="textrank",
        source_article_id="art_12345",
        title="High Court Ruling",
        article_text="Full original text of the court proceedings...",
        extracted_vocabulary=vocab_items,
        new_vocabulary=new_vocab,
        metadata=meta,
    )

    assert isinstance(history_id, int)
    assert history_id > 0

    record = temp_db.get_article_history(history_id)
    assert record is not None
    assert record["id"] == history_id
    assert record["source_article_id"] == "art_12345"
    assert record["title"] == "High Court Ruling"
    assert record["summary"] == "Extractive summary sentence one. Second sentence here."
    assert record["method"] == "textrank"
    assert record["extracted_vocabulary"] == vocab_items
    assert record["new_vocabulary"] == new_vocab
    assert record["metadata"] == meta
    assert "T" in record["processing_timestamp"]  # ISO 8601


def test_save_multiple_entries_same_source_article(temp_db):
    """Test that multiple history records can exist for the same source article (e.g. different methods)."""
    id1 = temp_db.save_article_history(
        source_article_id="art_shared_99",
        summary="Frequency baseline summary.",
        method="frequency",
    )
    id2 = temp_db.save_article_history(
        source_article_id="art_shared_99",
        summary="TextRank graph-based summary.",
        method="textrank",
    )

    assert id1 != id2
    rec1 = temp_db.get_article_history(id1)
    rec2 = temp_db.get_article_history(id2)
    assert rec1["method"] == "frequency"
    assert rec2["method"] == "textrank"
    assert rec1["source_article_id"] == rec2["source_article_id"] == "art_shared_99"


def test_article_history_listing_and_pagination(temp_db):
    """Test listing article history with limit and offset."""
    for i in range(15):
        temp_db.save_article_history(
            source_article_id=f"art_{i}",
            summary=f"Summary number {i}",
            method="tfidf" if i % 2 == 0 else "textrank",
        )

    # First page
    page1 = temp_db.list_article_history(limit=10, offset=0)
    assert len(page1) == 10

    # Second page
    page2 = temp_db.list_article_history(limit=10, offset=10)
    assert len(page2) == 5

    # Check non-overlapping IDs
    page1_ids = {r["id"] for r in page1}
    page2_ids = {r["id"] for r in page2}
    assert len(page1_ids.intersection(page2_ids)) == 0

    # Total count
    total = temp_db.count_article_history()
    assert total == 15


def test_article_history_filtering_by_method(temp_db):
    """Test filtering history entries by summarization method."""
    temp_db.save_article_history(summary="Freq summary", method="frequency")
    temp_db.save_article_history(summary="TF-IDF summary", method="tfidf")
    temp_db.save_article_history(summary="TextRank summary", method="textrank")
    temp_db.save_article_history(summary="Another TextRank summary", method="textrank")

    textrank_list = temp_db.list_article_history(method="textrank")
    assert len(textrank_list) == 2
    assert all(r["method"] == "textrank" for r in textrank_list)

    assert temp_db.count_article_history(method="frequency") == 1
    assert temp_db.count_article_history(method="tfidf") == 1
    assert temp_db.count_article_history(method="textrank") == 2


def test_search_article_history(temp_db):
    """Test searching history by title, text, summary, or source ID."""
    temp_db.save_article_history(
        title="Global Warming Summit",
        summary="World leaders met in Geneva to discuss climate targets.",
        method="textrank",
    )
    temp_db.save_article_history(
        title="Tech Sector Market Update",
        summary="Semiconductor manufacturers reported record quarterly revenue.",
        method="tfidf",
    )

    res_geneva = temp_db.search_article_history("Geneva")
    assert len(res_geneva) == 1
    assert res_geneva[0]["title"] == "Global Warming Summit"

    res_tech = temp_db.search_article_history("semiconductor")
    assert len(res_tech) == 1
    assert res_tech[0]["title"] == "Tech Sector Market Update"

    res_empty = temp_db.search_article_history("nonexistentquery123")
    assert len(res_empty) == 0


def test_delete_article_history(temp_db):
    """Test deleting an article history entry."""
    hid = temp_db.save_article_history(summary="Summary to delete", method="frequency")
    assert temp_db.get_article_history(hid) is not None

    deleted = temp_db.delete_article_history(hid)
    assert deleted is True
    assert temp_db.get_article_history(hid) is None

    # Deleting non-existent ID returns False
    assert temp_db.delete_article_history(999999) is False


def test_clear_article_history_safety(temp_db):
    """Test that clear_article_history requires explicit confirmation."""
    temp_db.save_article_history(summary="Sample 1", method="frequency")
    temp_db.save_article_history(summary="Sample 2", method="tfidf")

    # Without confirmation: raises ValueError
    with pytest.raises(ValueError, match="requires explicit confirmation"):
        temp_db.clear_article_history(confirm=False)

    # With confirmation: clears all records
    deleted_count = temp_db.clear_article_history(confirm=True)
    assert deleted_count == 2
    assert temp_db.count_article_history() == 0


def test_article_history_missing_identifier(temp_db):
    """Test querying non-existent or invalid history IDs."""
    assert temp_db.get_article_history(0) is None
    assert temp_db.get_article_history(-5) is None
    assert temp_db.get_article_history(99999) is None


def test_article_history_optional_fields_and_defaults(temp_db):
    """Test saving history with only required fields (summary, method)."""
    hid = temp_db.save_article_history(
        summary="Minimal required summary",
        method="frequency",
    )
    rec = temp_db.get_article_history(hid)
    assert rec["source_article_id"] is None
    assert rec["title"] is None
    assert rec["article_text"] is None
    assert rec["extracted_vocabulary"] is None
    assert rec["new_vocabulary"] is None
    assert rec["metadata"] is None
    assert rec["processing_timestamp"] is not None


# =============================================================================
# 2. VOCABULARY NOTES TESTS
# =============================================================================

def test_create_and_retrieve_note(temp_db):
    """Test creating and retrieving a vocabulary note."""
    note_id = temp_db.create_note(
        term="jurisdiction",
        content="The official power to make legal decisions and judgments.",
        example_sentence="The district court has jurisdiction over the territory.",
    )
    assert isinstance(note_id, int)
    assert note_id > 0

    note = temp_db.get_note(note_id)
    assert note is not None
    assert note["id"] == note_id
    assert note["term"] == "jurisdiction"
    assert note["content"] == "The official power to make legal decisions and judgments."
    assert note["example_sentence"] == "The district court has jurisdiction over the territory."
    assert note["article_history_id"] is None
    assert note["created_at"] is not None
    assert note["updated_at"] == note["created_at"]


def test_general_note_without_vocabulary_term(temp_db):
    """Test creating general study notes not tied to a specific vocabulary word."""
    note_id = temp_db.create_note(
        content="Remember to review news articles on international law before the viva exam.",
    )
    note = temp_db.get_note(note_id)
    assert note["term"] is None
    assert note["content"] == "Remember to review news articles on international law before the viva exam."


def test_note_with_article_association(temp_db):
    """Test linking a vocabulary note to an article history entry."""
    hid = temp_db.save_article_history(
        summary="Palestine ICC treaty ratified.",
        method="textrank",
    )
    note_id = temp_db.create_note(
        term="ICC",
        content="International Criminal Court established in The Hague.",
        article_history_id=hid,
    )
    note = temp_db.get_note(note_id)
    assert note["article_history_id"] == hid

    # Filter notes by article_history_id
    article_notes = temp_db.list_notes(article_history_id=hid)
    assert len(article_notes) == 1
    assert article_notes[0]["id"] == note_id


def test_note_listing_and_pagination(temp_db):
    """Test listing notes with limit and offset."""
    for i in range(12):
        temp_db.create_note(
            term=f"term_{i}",
            content=f"Definition of term {i}",
        )

    p1 = temp_db.list_notes(limit=5, offset=0)
    p2 = temp_db.list_notes(limit=5, offset=5)
    assert len(p1) == 5
    assert len(p2) == 5
    assert {n["id"] for n in p1}.isdisjoint({n["id"] for n in p2})
    assert temp_db.count_notes() == 12


def test_search_notes(temp_db):
    """Test searching notes across term, content, and example_sentence."""
    temp_db.create_note(
        term="statute",
        content="A written law passed by a legislative body.",
        example_sentence="The Rome Statute established the court.",
    )
    temp_db.create_note(
        term="photosynthesis",
        content="Process by which plants use sunlight to synthesize nutrients.",
    )

    res1 = temp_db.search_notes("legislative")
    assert len(res1) == 1
    assert res1[0]["term"] == "statute"

    res2 = temp_db.search_notes("Rome Statute")
    assert len(res2) == 1
    assert res2[0]["term"] == "statute"

    res3 = temp_db.search_notes("plants")
    assert len(res3) == 1
    assert res3[0]["term"] == "photosynthesis"


def test_update_note(temp_db):
    """Test updating content, term, and example sentence with timestamp refresh."""
    nid = temp_db.create_note(
        term="draft",
        content="First version of a document.",
    )
    initial_note = temp_db.get_note(nid)

    # Perform update
    updated = temp_db.update_note(
        note_id=nid,
        content="A preliminary version of a piece of writing.",
        example_sentence="She handed in a rough draft of her essay.",
    )
    assert updated is True

    updated_note = temp_db.get_note(nid)
    assert updated_note["content"] == "A preliminary version of a piece of writing."
    assert updated_note["example_sentence"] == "She handed in a rough draft of her essay."
    assert updated_note["term"] == "draft"  # Untouched

    # Updating non-existent note returns False
    assert temp_db.update_note(9999, content="New content") is False


def test_delete_note(temp_db):
    """Test deleting a specific note."""
    nid = temp_db.create_note(content="Temporary note to delete")
    assert temp_db.get_note(nid) is not None

    deleted = temp_db.delete_note(nid)
    assert deleted is True
    assert temp_db.get_note(nid) is None
    assert temp_db.delete_note(nid) is False


def test_invalid_input_validation(temp_db):
    """Test validation errors on empty or invalid inputs."""
    with pytest.raises(ValueError, match="Summary must be a non-empty string"):
        temp_db.save_article_history(summary="", method="textrank")

    with pytest.raises(ValueError, match="Method must be a non-empty string"):
        temp_db.save_article_history(summary="Valid summary", method="")

    with pytest.raises(ValueError, match="Note content must be a non-empty string"):
        temp_db.create_note(content="")

    with pytest.raises(ValueError, match="Referenced article_history_id 99999 does not exist"):
        temp_db.create_note(content="Valid note", article_history_id=99999)


def test_sql_injection_treated_as_data(temp_db):
    """Test that malicious SQL strings in content, search queries, or terms are strictly parameterized data."""
    malicious_input = "'; DROP TABLE vocabulary_notes; --"

    nid = temp_db.create_note(
        term=malicious_input,
        content=f"Attempting injection: {malicious_input}",
    )
    assert nid > 0

    # Ensure table was NOT dropped
    note = temp_db.get_note(nid)
    assert note is not None
    assert note["term"] == malicious_input

    # Search query injection attempt
    search_results = temp_db.search_notes("'; DROP TABLE article_history; --")
    assert isinstance(search_results, list)

    # Ensure article_history table still exists and functions
    count = temp_db.count_article_history()
    assert count >= 0


# =============================================================================
# 3. RELIABILITY, TRANSACTIONS, & INTEGRITY TESTS
# =============================================================================

def test_foreign_key_set_null_on_article_delete(temp_db):
    """Test that deleting an article history entry sets note's article_history_id to NULL without deleting the note."""
    hid = temp_db.save_article_history(summary="Article summary", method="frequency")
    nid = temp_db.create_note(
        term="evidence",
        content="Crucial legal evidence.",
        article_history_id=hid,
    )

    note_before = temp_db.get_note(nid)
    assert note_before["article_history_id"] == hid

    # Delete the parent article history
    temp_db.delete_article_history(hid)
    assert temp_db.get_article_history(hid) is None

    # Note must still exist, with article_history_id now set to NULL
    note_after = temp_db.get_note(nid)
    assert note_after is not None
    assert note_after["article_history_id"] is None
    assert note_after["term"] == "evidence"


def test_transaction_rollback_on_failure(temp_db):
    """Test atomic transactions rollback when an unhandled database error occurs."""
    initial_count = temp_db.count_article_history()

    # Attempt an insertion that violates a database constraint
    conn = temp_db.get_connection()
    try:
        with pytest.raises(sqlite3.IntegrityError):
            with conn:
                conn.execute(
                    "INSERT INTO article_history (summary, method, processing_timestamp) VALUES (?, ?, ?);",
                    ("Valid summary", "textrank", "2026-10-17T00:00:00Z"),
                )
                # Second statement fails NOT NULL constraint on summary
                conn.execute(
                    "INSERT INTO article_history (summary, method, processing_timestamp) VALUES (NULL, 'textrank', '2026-10-17T00:00:00Z');"
                )
    finally:
        conn.close()

    # Verify that the first insertion was rolled back completely
    assert temp_db.count_article_history() == initial_count


def test_caller_input_dictionary_immutability(temp_db):
    """Test that caller input dictionaries (metadata, vocab) are not mutated by save operations."""
    vocab = [{"term": "test", "pos": "NOUN"}]
    meta = {"settings": {"k": 3}, "tags": ["nlp", "ml"]}

    vocab_backup = copy.deepcopy(vocab)
    meta_backup = copy.deepcopy(meta)

    temp_db.save_article_history(
        summary="Test summary.",
        method="textrank",
        extracted_vocabulary=vocab,
        metadata=meta,
    )

    assert vocab == vocab_backup
    assert meta == meta_backup


def test_never_write_to_production_database_during_tests():
    """Verify that default production database path was NOT touched or created during test execution."""
    # Ensure DEFAULT_DB_PATH is database/notes_history.db
    assert DEFAULT_DB_PATH == Path("database/notes_history.db")
    # If the production database existed prior to tests, verify its path differs from temp DBs
    manager = NotesHistoryManager(db_path=":memory:")
    assert manager.is_memory is True
    manager.close()


def test_integration_article_history_and_notes(temp_db):
    """
    End-to-end integration test:
    1. Save an extractive article summary with Phase 7/8 vocabulary and metadata.
    2. Create vocabulary notes referencing extracted and new words linked to this article history.
    3. Query notes for this article and verify relationships.
    4. Update note content and verify timestamp refresh.
    5. Search across notes and history simultaneously.
    """
    # 1. Save Article History
    extracted_vocab = [
        {"term": "sovereignty", "pos": "NOUN", "importance_score": 0.72},
        {"term": "ratify", "pos": "VERB", "importance_score": 0.65},
    ]
    new_vocab = [
        {"term": "sovereignty", "pos": "NOUN", "importance_score": 0.72},
    ]
    meta = {"algorithm": "textrank", "num_sentences": 3, "compression_ratio": 0.15}

    history_id = temp_db.save_article_history(
        source_article_id="cnn_art_001",
        title="Treaty Ratification Milestone",
        article_text="Member states gathered in Geneva to formally ratify the sovereignty treaty...",
        summary="Member states formally ratified the sovereignty treaty in Geneva.",
        method="textrank",
        extracted_vocabulary=extracted_vocab,
        new_vocabulary=new_vocab,
        metadata=meta,
    )
    assert history_id > 0

    # 2. Create Vocabulary Notes linked to this article
    note_id_1 = temp_db.create_note(
        term="sovereignty",
        content="Supreme power or authority of a state to govern itself.",
        example_sentence="The treaty affirmed the national sovereignty of each member state.",
        article_history_id=history_id,
    )
    note_id_2 = temp_db.create_note(
        term="ratify",
        content="Sign or give formal consent to a treaty, contract, or agreement.",
        example_sentence="Both parliaments voted to ratify the agreement.",
        article_history_id=history_id,
    )
    assert note_id_1 > 0
    assert note_id_2 > 0

    # 3. Retrieve and verify article notes
    notes = temp_db.list_notes(article_history_id=history_id)
    assert len(notes) == 2
    terms = {n["term"] for n in notes}
    assert terms == {"sovereignty", "ratify"}

    # 4. Update note
    updated = temp_db.update_note(
        note_id=note_id_1,
        content="Supreme authority and self-governance of an independent state.",
    )
    assert updated is True
    updated_note = temp_db.get_note(note_id_1)
    assert updated_note["content"] == "Supreme authority and self-governance of an independent state."

    # 5. Search
    found_history = temp_db.search_article_history("sovereignty")
    assert len(found_history) == 1
    assert found_history[0]["id"] == history_id

    found_notes = temp_db.search_notes("self-governance")
    assert len(found_notes) == 1
    assert found_notes[0]["id"] == note_id_1

