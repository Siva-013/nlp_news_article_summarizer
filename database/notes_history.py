"""
Notes and Article History Persistence Module
Part of Phase 9: Notes and Article History

This module provides a local, robust SQLite persistence layer for:
1. Article History: Saving, retrieving, listing, searching, filtering, and deleting
   previously summarized articles and associated Phase 7/8 vocabulary metadata.
2. Vocabulary Notes: Creating, retrieving, listing, searching, updating, and deleting
   learner vocabulary notes and example sentences with optional article history linkage.

Key Architectural Guarantees:
- Strict Classical Implementation: Standard library sqlite3 only; zero external ORMs.
- SQL Injection Protection: All operations strictly use parameterized queries.
- Data Integrity: Foreign-key support with ON DELETE SET NULL preserves user notes
  even if the parent article history entry is removed.
- Transaction Safety: Atomic transactions with automatic rollback on errors.
- Non-Mutating: Never modifies caller-supplied dictionaries or inputs.
- Test Isolation: Supports in-memory (":memory:") and file-based SQLite databases.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union


DEFAULT_DB_PATH = Path("database/notes_history.db")


def get_utc_now_iso() -> str:
    """Generate canonical ISO 8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


class NotesHistoryManager:
    """
    Persistence manager for Article History and Vocabulary Notes using SQLite.
    """

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        auto_init: bool = True,
    ):
        """
        Initialize the persistence manager.

        Args:
            db_path: Path to SQLite database file or ':memory:'.
                     Defaults to 'database/notes_history.db'.
            auto_init: Automatically create schema and indexes if True.
        """
        if db_path is None:
            self.db_path = str(DEFAULT_DB_PATH)
        elif isinstance(db_path, Path):
            self.db_path = str(db_path)
        else:
            self.db_path = str(db_path)

        self.is_memory = (self.db_path == ":memory:")
        self._memory_conn: Optional[sqlite3.Connection] = None

        # Ensure parent directory exists for file-based database
        if not self.is_memory:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        else:
            # For in-memory database, preserve a single connection across operations
            self._memory_conn = sqlite3.connect(":memory:")
            self._memory_conn.row_factory = sqlite3.Row
            self._memory_conn.execute("PRAGMA foreign_keys = ON;")

        if auto_init:
            self.initialize_schema()

    def get_connection(self) -> sqlite3.Connection:
        """
        Get a SQLite connection with foreign keys enabled and row_factory configured.
        """
        if self.is_memory and self._memory_conn is not None:
            return self._memory_conn

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def close(self) -> None:
        """Close database connection (primarily relevant for in-memory databases)."""
        if self._memory_conn is not None:
            self._memory_conn.close()
            self._memory_conn = None

    def __enter__(self) -> "NotesHistoryManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def initialize_schema(self) -> None:
        """
        Create tables and indexes if they do not already exist.
        """
        conn = self.get_connection()
        try:
            with conn:
                # 1. Article History table
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS article_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_article_id TEXT,
                        title TEXT,
                        article_text TEXT,
                        summary TEXT NOT NULL,
                        method TEXT NOT NULL,
                        processing_timestamp TEXT NOT NULL,
                        extracted_vocabulary TEXT,
                        new_vocabulary TEXT,
                        metadata TEXT
                    );
                    """
                )

                # Indexes for article_history
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_ah_source_id ON article_history(source_article_id);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_ah_method ON article_history(method);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_ah_timestamp ON article_history(processing_timestamp DESC);"
                )

                # 2. Vocabulary Notes table
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS vocabulary_notes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        term TEXT,
                        content TEXT NOT NULL,
                        example_sentence TEXT,
                        article_history_id INTEGER,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY(article_history_id) REFERENCES article_history(id) ON DELETE SET NULL
                    );
                    """
                )

                # Indexes for vocabulary_notes
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_notes_term ON vocabulary_notes(term);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_notes_article_id ON vocabulary_notes(article_history_id);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_notes_created_at ON vocabulary_notes(created_at DESC);"
                )
        finally:
            if not self.is_memory:
                conn.close()

    # =========================================================================
    # ARTICLE HISTORY OPERATIONS
    # =========================================================================

    def save_article_history(
        self,
        summary: str,
        method: str,
        source_article_id: Optional[str] = None,
        title: Optional[str] = None,
        article_text: Optional[str] = None,
        processing_timestamp: Optional[str] = None,
        extracted_vocabulary: Optional[Union[List[Any], Dict[str, Any], str]] = None,
        new_vocabulary: Optional[Union[List[Any], Dict[str, Any], str]] = None,
        metadata: Optional[Union[Dict[str, Any], str]] = None,
    ) -> int:
        """
        Save an article summarization record and its associated vocabulary into history.

        Args:
            summary: Extractive summary text (required, non-empty).
            method: Summarization method (e.g. 'textrank', 'tfidf', 'frequency').
            source_article_id: Original dataset or source article ID.
            title: Headline or article title.
            article_text: Original article text or excerpt.
            processing_timestamp: ISO 8601 UTC timestamp string (generated if omitted).
            extracted_vocabulary: Vocabulary items from Phase 7 (list, dict, or JSON string).
            new_vocabulary: New vocabulary items from Phase 8 (list, dict, or JSON string).
            metadata: Evaluation metrics, parameters, or metadata (dict or JSON string).

        Returns:
            The unique integer history ID of the newly saved record.
        """
        if not summary or not isinstance(summary, str) or not summary.strip():
            raise ValueError("Summary must be a non-empty string.")
        if not method or not isinstance(method, str) or not method.strip():
            raise ValueError("Method must be a non-empty string.")

        timestamp = processing_timestamp or get_utc_now_iso()

        # Serialize complex data structures safely without mutating inputs
        vocab_json = (
            json.dumps(extracted_vocabulary, ensure_ascii=False)
            if isinstance(extracted_vocabulary, (list, dict))
            else extracted_vocabulary
        )
        new_vocab_json = (
            json.dumps(new_vocabulary, ensure_ascii=False)
            if isinstance(new_vocabulary, (list, dict))
            else new_vocabulary
        )
        meta_json = (
            json.dumps(metadata, ensure_ascii=False)
            if isinstance(metadata, (list, dict))
            else metadata
        )

        conn = self.get_connection()
        try:
            with conn:
                cursor = conn.execute(
                    """
                    INSERT INTO article_history (
                        source_article_id,
                        title,
                        article_text,
                        summary,
                        method,
                        processing_timestamp,
                        extracted_vocabulary,
                        new_vocabulary,
                        metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        source_article_id,
                        title,
                        article_text,
                        summary.strip(),
                        method.strip().lower(),
                        timestamp,
                        vocab_json,
                        new_vocab_json,
                        meta_json,
                    ),
                )
                return int(cursor.lastrowid)
        finally:
            if not self.is_memory:
                conn.close()

    def get_article_history(self, history_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve an article history entry by its unique history ID.

        Args:
            history_id: Unique integer primary key.

        Returns:
            Dictionary with history fields and deserialized JSON structures, or None.
        """
        if not isinstance(history_id, int) or history_id <= 0:
            return None

        conn = self.get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT id, source_article_id, title, article_text, summary,
                       method, processing_timestamp, extracted_vocabulary,
                       new_vocabulary, metadata
                FROM article_history
                WHERE id = ?;
                """,
                (history_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return self._format_article_history_row(row)
        finally:
            if not self.is_memory:
                conn.close()

    def list_article_history(
        self,
        limit: int = 20,
        offset: int = 0,
        method: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List article history records with pagination and optional method filtering.

        Args:
            limit: Maximum number of records to return.
            offset: Number of records to skip.
            method: Filter by summarization method (e.g. 'textrank', 'tfidf', 'frequency').

        Returns:
            List of formatted history entry dictionaries.
        """
        limit = max(1, int(limit))
        offset = max(0, int(offset))

        conn = self.get_connection()
        try:
            if method:
                cursor = conn.execute(
                    """
                    SELECT id, source_article_id, title, article_text, summary,
                           method, processing_timestamp, extracted_vocabulary,
                           new_vocabulary, metadata
                    FROM article_history
                    WHERE method = ?
                    ORDER BY processing_timestamp DESC, id DESC
                    LIMIT ? OFFSET ?;
                    """,
                    (method.strip().lower(), limit, offset),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT id, source_article_id, title, article_text, summary,
                           method, processing_timestamp, extracted_vocabulary,
                           new_vocabulary, metadata
                    FROM article_history
                    ORDER BY processing_timestamp DESC, id DESC
                    LIMIT ? OFFSET ?;
                    """,
                    (limit, offset),
                )

            return [self._format_article_history_row(row) for row in cursor.fetchall()]
        finally:
            if not self.is_memory:
                conn.close()

    def count_article_history(self, method: Optional[str] = None) -> int:
        """
        Count total article history records matching filter.
        """
        conn = self.get_connection()
        try:
            if method:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM article_history WHERE method = ?;",
                    (method.strip().lower(),),
                )
            else:
                cursor = conn.execute("SELECT COUNT(*) FROM article_history;")
            return int(cursor.fetchone()[0])
        finally:
            if not self.is_memory:
                conn.close()

    def search_article_history(
        self,
        query: str,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Search article history across title, article_text, summary, and source_article_id.
        Uses safe SQL parameterized LIKE binding.

        Args:
            query: Search string.
            limit: Pagination limit.
            offset: Pagination offset.

        Returns:
            List of matching article history dictionaries.
        """
        cleaned_query = str(query).strip()
        if not cleaned_query:
            return []

        limit = max(1, int(limit))
        offset = max(0, int(offset))
        param = f"%{cleaned_query}%"

        conn = self.get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT id, source_article_id, title, article_text, summary,
                       method, processing_timestamp, extracted_vocabulary,
                       new_vocabulary, metadata
                FROM article_history
                WHERE title LIKE ? OR article_text LIKE ? OR summary LIKE ? OR source_article_id LIKE ?
                ORDER BY processing_timestamp DESC, id DESC
                LIMIT ? OFFSET ?;
                """,
                (param, param, param, param, limit, offset),
            )
            return [self._format_article_history_row(row) for row in cursor.fetchall()]
        finally:
            if not self.is_memory:
                conn.close()

    def count_search_article_history(self, query: str) -> int:
        """
        Count total article history records matching the search query
        (title, article_text, summary, source_article_id), so search
        pagination can report a true total.
        """
        cleaned_query = str(query).strip()
        if not cleaned_query:
            return 0
        param = f"%{cleaned_query}%"
        conn = self.get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM article_history
                WHERE title LIKE ? OR article_text LIKE ? OR summary LIKE ? OR source_article_id LIKE ?;
                """,
                (param, param, param, param),
            )
            return int(cursor.fetchone()[0])
        finally:
            if not self.is_memory:
                conn.close()

    def delete_article_history(self, history_id: int) -> bool:
        """
        Delete a single article history entry by ID.

        Args:
            history_id: Unique history ID to delete.

        Returns:
            True if an entry was deleted, False if not found.
        """
        if not isinstance(history_id, int) or history_id <= 0:
            return False

        conn = self.get_connection()
        try:
            with conn:
                cursor = conn.execute(
                    "DELETE FROM article_history WHERE id = ?;",
                    (history_id,),
                )
                return cursor.rowcount > 0
        finally:
            if not self.is_memory:
                conn.close()

    def clear_article_history(self, confirm: bool = False) -> int:
        """
        Delete all article history records.
        Requires explicit confirm=True to prevent accidental data loss.

        Args:
            confirm: Must be explicitly True.

        Returns:
            Total count of deleted records.
        """
        if confirm is not True:
            raise ValueError(
                "Clearing article history requires explicit confirmation: confirm=True."
            )

        conn = self.get_connection()
        try:
            with conn:
                cursor = conn.execute("DELETE FROM article_history;")
                return cursor.rowcount
        finally:
            if not self.is_memory:
                conn.close()

    @staticmethod
    def _format_article_history_row(row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a sqlite3.Row for article_history into a dictionary with deserialized JSON fields."""
        def safe_json_loads(val: Optional[str]) -> Any:
            if not val:
                return None
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return val

        return {
            "id": row["id"],
            "source_article_id": row["source_article_id"],
            "title": row["title"],
            "article_text": row["article_text"],
            "summary": row["summary"],
            "method": row["method"],
            "processing_timestamp": row["processing_timestamp"],
            "extracted_vocabulary": safe_json_loads(row["extracted_vocabulary"]),
            "new_vocabulary": safe_json_loads(row["new_vocabulary"]),
            "metadata": safe_json_loads(row["metadata"]),
        }

    # =========================================================================
    # VOCABULARY NOTES OPERATIONS
    # =========================================================================

    def create_note(
        self,
        content: str,
        term: Optional[str] = None,
        example_sentence: Optional[str] = None,
        article_history_id: Optional[int] = None,
        created_at: Optional[str] = None,
    ) -> int:
        """
        Create a new vocabulary or general reading note.

        Args:
            content: Note text content (required, non-empty).
            term: Associated vocabulary term or phrase (optional).
            example_sentence: Example usage sentence (optional).
            article_history_id: Associated article history entry ID (optional).
            created_at: ISO 8601 UTC timestamp string (generated if omitted).

        Returns:
            The unique integer ID of the created note.
        """
        if not content or not isinstance(content, str) or not content.strip():
            raise ValueError("Note content must be a non-empty string.")

        timestamp = created_at or get_utc_now_iso()
        cleaned_term = term.strip() if term and isinstance(term, str) and term.strip() else None
        cleaned_example = (
            example_sentence.strip()
            if example_sentence and isinstance(example_sentence, str) and example_sentence.strip()
            else None
        )

        conn = self.get_connection()
        try:
            with conn:
                # If article_history_id is supplied, verify it exists to ensure integrity
                if article_history_id is not None:
                    check = conn.execute(
                        "SELECT id FROM article_history WHERE id = ?;",
                        (article_history_id,),
                    ).fetchone()
                    if not check:
                        raise ValueError(
                            f"Referenced article_history_id {article_history_id} does not exist."
                        )

                cursor = conn.execute(
                    """
                    INSERT INTO vocabulary_notes (
                        term,
                        content,
                        example_sentence,
                        article_history_id,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?);
                    """,
                    (
                        cleaned_term,
                        content.strip(),
                        cleaned_example,
                        article_history_id,
                        timestamp,
                        timestamp,
                    ),
                )
                return int(cursor.lastrowid)
        finally:
            if not self.is_memory:
                conn.close()

    def get_note(self, note_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a note by its ID.

        Args:
            note_id: Unique integer note ID.

        Returns:
            Dictionary with note fields, or None if not found.
        """
        if not isinstance(note_id, int) or note_id <= 0:
            return None

        conn = self.get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT id, term, content, example_sentence, article_history_id,
                       created_at, updated_at
                FROM vocabulary_notes
                WHERE id = ?;
                """,
                (note_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return self._format_note_row(row)
        finally:
            if not self.is_memory:
                conn.close()

    def list_notes(
        self,
        limit: int = 20,
        offset: int = 0,
        term: Optional[str] = None,
        article_history_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        List notes with pagination and optional filtering by term or article.

        Args:
            limit: Maximum notes to return.
            offset: Notes to skip.
            term: Filter by exact vocabulary term (case-insensitive).
            article_history_id: Filter by associated article history ID.

        Returns:
            List of note dictionaries ordered by created_at DESC, id DESC.
        """
        limit = max(1, int(limit))
        offset = max(0, int(offset))

        conditions = []
        params: List[Any] = []

        if term and isinstance(term, str) and term.strip():
            conditions.append("LOWER(term) = LOWER(?)")
            params.append(term.strip())

        if article_history_id is not None:
            conditions.append("article_history_id = ?")
            params.append(article_history_id)

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        sql = f"""
            SELECT id, term, content, example_sentence, article_history_id,
                   created_at, updated_at
            FROM vocabulary_notes
            {where_clause}
            ORDER BY created_at DESC, id DESC
            LIMIT ? OFFSET ?;
        """
        params.extend([limit, offset])

        conn = self.get_connection()
        try:
            cursor = conn.execute(sql, tuple(params))
            return [self._format_note_row(row) for row in cursor.fetchall()]
        finally:
            if not self.is_memory:
                conn.close()

    def count_notes(
        self,
        term: Optional[str] = None,
        article_history_id: Optional[int] = None,
    ) -> int:
        """Count total notes matching filter."""
        conditions = []
        params: List[Any] = []

        if term and isinstance(term, str) and term.strip():
            conditions.append("LOWER(term) = LOWER(?)")
            params.append(term.strip())

        if article_history_id is not None:
            conditions.append("article_history_id = ?")
            params.append(article_history_id)

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = f"SELECT COUNT(*) FROM vocabulary_notes {where_clause};"

        conn = self.get_connection()
        try:
            cursor = conn.execute(sql, tuple(params))
            return int(cursor.fetchone()[0])
        finally:
            if not self.is_memory:
                conn.close()

    def search_notes(
        self,
        query: str,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Search notes across content, term, and example_sentence.
        Uses safe SQL parameterized LIKE binding.

        Args:
            query: Search query string.
            limit: Pagination limit.
            offset: Pagination offset.

        Returns:
            List of matching note dictionaries.
        """
        cleaned = str(query).strip()
        if not cleaned:
            return []

        limit = max(1, int(limit))
        offset = max(0, int(offset))
        param = f"%{cleaned}%"

        conn = self.get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT id, term, content, example_sentence, article_history_id,
                       created_at, updated_at
                FROM vocabulary_notes
                WHERE content LIKE ? OR term LIKE ? OR example_sentence LIKE ?
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?;
                """,
                (param, param, param, limit, offset),
            )
            return [self._format_note_row(row) for row in cursor.fetchall()]
        finally:
            if not self.is_memory:
                conn.close()

    def count_search_notes(self, query: str) -> int:
        """
        Count total notes matching the search query
        (content, term, example_sentence), so search pagination
        can report a true total.
        """
        cleaned = str(query).strip()
        if not cleaned:
            return 0
        param = f"%{cleaned}%"
        conn = self.get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM vocabulary_notes
                WHERE content LIKE ? OR term LIKE ? OR example_sentence LIKE ?;
                """,
                (param, param, param),
            )
            return int(cursor.fetchone()[0])
        finally:
            if not self.is_memory:
                conn.close()

    def update_note(
        self,
        note_id: int,
        content: Optional[str] = None,
        term: Optional[str] = None,
        example_sentence: Optional[str] = None,
        article_history_id: Optional[int] = None,
        clear_term: bool = False,
        clear_example: bool = False,
        clear_article_history: bool = False,
        updated_at: Optional[str] = None,
    ) -> bool:
        """
        Update fields of an existing note. Automatically refreshes updated_at timestamp.

        Args:
            note_id: ID of the note to update.
            content: New note content (optional).
            term: New vocabulary term (optional).
            example_sentence: New example sentence (optional).
            article_history_id: New associated article history ID (optional).
            clear_term: Explicitly set term to NULL.
            clear_example: Explicitly set example_sentence to NULL.
            clear_article_history: Explicitly set article_history_id to NULL.
            updated_at: Optional custom ISO timestamp.

        Returns:
            True if the note was found and updated, False if note_id does not exist.
        """
        if not isinstance(note_id, int) or note_id <= 0:
            return False

        updates = []
        params: List[Any] = []

        if content is not None:
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Updated content must be a non-empty string.")
            updates.append("content = ?")
            params.append(content.strip())

        if clear_term:
            updates.append("term = NULL")
        elif term is not None:
            cleaned_term = term.strip() if isinstance(term, str) and term.strip() else None
            updates.append("term = ?")
            params.append(cleaned_term)

        if clear_example:
            updates.append("example_sentence = NULL")
        elif example_sentence is not None:
            cleaned_ex = (
                example_sentence.strip()
                if isinstance(example_sentence, str) and example_sentence.strip()
                else None
            )
            updates.append("example_sentence = ?")
            params.append(cleaned_ex)

        if clear_article_history:
            updates.append("article_history_id = NULL")
        elif article_history_id is not None:
            updates.append("article_history_id = ?")
            params.append(article_history_id)

        # Always update updated_at
        timestamp = updated_at or get_utc_now_iso()
        updates.append("updated_at = ?")
        params.append(timestamp)

        params.append(note_id)

        conn = self.get_connection()
        try:
            with conn:
                # If article_history_id is provided, verify it exists
                if article_history_id is not None and not clear_article_history:
                    check = conn.execute(
                        "SELECT id FROM article_history WHERE id = ?;",
                        (article_history_id,),
                    ).fetchone()
                    if not check:
                        raise ValueError(
                            f"Referenced article_history_id {article_history_id} does not exist."
                        )

                sql = f"UPDATE vocabulary_notes SET {', '.join(updates)} WHERE id = ?;"
                cursor = conn.execute(sql, tuple(params))
                return cursor.rowcount > 0
        finally:
            if not self.is_memory:
                conn.close()

    def delete_note(self, note_id: int) -> bool:
        """
        Delete a single note by ID.

        Args:
            note_id: Note ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        if not isinstance(note_id, int) or note_id <= 0:
            return False

        conn = self.get_connection()
        try:
            with conn:
                cursor = conn.execute(
                    "DELETE FROM vocabulary_notes WHERE id = ?;",
                    (note_id,),
                )
                return cursor.rowcount > 0
        finally:
            if not self.is_memory:
                conn.close()

    def clear_notes(self, confirm: bool = False) -> int:
        """
        Delete all vocabulary notes.
        Requires explicit confirm=True to prevent accidental data loss.
        """
        if confirm is not True:
            raise ValueError(
                "Clearing vocabulary notes requires explicit confirmation: confirm=True."
            )

        conn = self.get_connection()
        try:
            with conn:
                cursor = conn.execute("DELETE FROM vocabulary_notes;")
                return cursor.rowcount
        finally:
            if not self.is_memory:
                conn.close()

    @staticmethod
    def _format_note_row(row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a sqlite3.Row for vocabulary_notes into a dictionary."""
        return {
            "id": row["id"],
            "term": row["term"],
            "content": row["content"],
            "example_sentence": row["example_sentence"],
            "article_history_id": row["article_history_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
