"""
Pydantic Request and Response Schemas
Phase 11: FastAPI Backend Integration
"""

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# HEALTH & METADATA SCHEMAS
# =============================================================================

class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "healthy"})
    database_connected: bool = Field(..., json_schema_extra={"example": True})
    version: str = Field(..., json_schema_extra={"example": "1.0.0"})
    engine: str = Field(..., json_schema_extra={"example": "classical_nlp_fastapi"})


class RootResponse(BaseModel):
    name: str
    version: str
    status: str
    docs_url: str
    redoc_url: str
    endpoints: Dict[str, str]


# =============================================================================
# SUMMARIZATION SCHEMAS
# =============================================================================

SummarizationMethod = Literal["frequency", "tfidf", "textrank"]


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw news article text to summarize.")
    method: SummarizationMethod = Field(
        default="frequency",
        description="Extractive summarization algorithm: 'frequency', 'tfidf', or 'textrank'."
    )
    num_sentences: int = Field(
        default=3,
        ge=1,
        le=50,
        description="Number of sentences to extract (default: 3)."
    )
    article_id: Optional[str] = Field(
        default="custom_article",
        description="Optional identifier tag for the article."
    )

    @field_validator("text")
    @classmethod
    def validate_non_empty_text(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Input text cannot be empty or contain only whitespace.")
        return cleaned


class SelectedSentence(BaseModel):
    sentence_index: int
    score: float
    original_text: str
    content_words: Optional[List[str]] = None


class SummarizeResponse(BaseModel):
    article_id: str
    method: str
    num_sentences_requested: int
    num_sentences_selected: int
    summary: str
    selected_sentences: List[SelectedSentence]
    highlights: Optional[str] = ""
    top_words: Optional[List[Dict[str, Any]]] = None
    rankings: Optional[List[Dict[str, Any]]] = None


# =============================================================================
# VOCABULARY EXTRACTION & NEW-WORD DETECTION SCHEMAS
# =============================================================================

class VocabularyExtractRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw news article text.")
    article_id: Optional[str] = Field(default="raw_article", description="Article ID.")
    max_items: Optional[int] = Field(default=30, ge=1, le=200, description="Maximum items to extract.")
    include_entities: Optional[bool] = Field(default=True, description="Whether to include named entities.")

    @field_validator("text")
    @classmethod
    def validate_non_empty_text(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Input text cannot be empty or contain only whitespace.")
        return cleaned


class VocabularyItem(BaseModel):
    term: str
    normalized_form: str
    lemma: str
    pos: str
    frequency: int
    sentence_coverage: float
    sentence_indices: List[int]
    entity_label: Optional[str] = None
    importance_score: float
    selection_reason: Optional[str] = None


class VocabularyExtractResponse(BaseModel):
    article_id: str
    method: str
    configuration: Dict[str, Any]
    total_candidate_items: int
    returned_item_count: int
    items: List[Dict[str, Any]]


class VocabularyDetectNewRequest(BaseModel):
    text: Optional[str] = Field(
        default=None,
        description="Raw article text (if provided, vocabulary is extracted automatically)."
    )
    extraction_record: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Pre-extracted Phase 7 extraction record dictionary."
    )
    known_vocabulary: List[Union[str, Dict[str, Any]]] = Field(
        default_factory=list,
        description="List of words known by the user (strings or dictionaries)."
    )
    max_items: Optional[int] = Field(
        default=30,
        ge=1,
        le=200,
        description="Maximum items to extract when text is supplied."
    )
    include_entities: Optional[bool] = Field(
        default=True,
        description="Whether to include named entities when text is supplied."
    )

    @field_validator("text")
    @classmethod
    def validate_text_if_present(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            cleaned = v.strip()
            if not cleaned:
                raise ValueError("Input text cannot be empty or contain only whitespace.")
            return cleaned
        return v


class VocabularyDetectNewResponse(BaseModel):
    article_id: str
    method: str
    matching_policy: str
    total_vocabulary_count: int
    known_word_count: int
    new_word_count: int
    known_words: List[Dict[str, Any]]
    new_words: List[Dict[str, Any]]
    validation: Dict[str, Any]


# =============================================================================
# ARTICLE HISTORY SCHEMAS
# =============================================================================

class ArticleHistoryCreateRequest(BaseModel):
    summary: str = Field(..., min_length=1, description="Summary text.")
    method: str = Field(..., min_length=1, description="Summarization method used.")
    title: Optional[str] = Field(default=None, description="Optional article title.")
    article_text: Optional[str] = Field(default=None, description="Full original article text.")
    source_article_id: Optional[str] = Field(default=None, description="Dataset or source ID.")
    extracted_vocabulary: Optional[Union[List[Any], Dict[str, Any]]] = Field(
        default=None,
        description="Phase 7 extracted vocabulary metadata."
    )
    new_vocabulary: Optional[Union[List[Any], Dict[str, Any]]] = Field(
        default=None,
        description="Phase 8 new vocabulary metadata."
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional custom metadata."
    )

    @field_validator("summary", "method")
    @classmethod
    def validate_non_empty_strings(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Field cannot be empty or contain only whitespace.")
        return cleaned


class ArticleHistoryResponse(BaseModel):
    id: int
    source_article_id: Optional[str] = None
    title: Optional[str] = None
    article_text: Optional[str] = None
    summary: str
    method: str
    processing_timestamp: str
    extracted_vocabulary: Optional[Any] = None
    new_vocabulary: Optional[Any] = None
    metadata: Optional[Any] = None


class ArticleHistoryListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: List[ArticleHistoryResponse]


# =============================================================================
# VOCABULARY NOTES SCHEMAS
# =============================================================================

class NoteCreateRequest(BaseModel):
    content: str = Field(..., min_length=1, description="Note body content.")
    term: Optional[str] = Field(default=None, description="Associated vocabulary term.")
    example_sentence: Optional[str] = Field(default=None, description="Example usage sentence.")
    article_history_id: Optional[int] = Field(default=None, description="Associated article history ID.")

    @field_validator("content")
    @classmethod
    def validate_content_non_empty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Note content cannot be empty or contain only whitespace.")
        return cleaned


class NoteUpdateRequest(BaseModel):
    content: Optional[str] = Field(default=None, description="Updated note content.")
    term: Optional[str] = Field(default=None, description="Updated vocabulary term.")
    example_sentence: Optional[str] = Field(default=None, description="Updated example sentence.")
    article_history_id: Optional[int] = Field(default=None, description="Updated article history ID.")
    clear_term: bool = Field(default=False, description="Set term to NULL.")
    clear_example: bool = Field(default=False, description="Set example_sentence to NULL.")
    clear_article_history: bool = Field(default=False, description="Set article_history_id to NULL.")


class NoteResponse(BaseModel):
    id: int
    term: Optional[str] = None
    content: str
    example_sentence: Optional[str] = None
    article_history_id: Optional[int] = None
    created_at: str
    updated_at: str


class NoteListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: List[NoteResponse]


# =============================================================================
# TEXT-TO-SPEECH (TTS) SCHEMAS
# =============================================================================

class SpeechSynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text summary to synthesize into speech.")
    rate: Optional[int] = Field(default=175, ge=50, le=400, description="Speech rate (50-400 wpm).")
    volume: Optional[float] = Field(default=1.0, ge=0.0, le=1.0, description="Volume level (0.0 to 1.0).")
    voice_id: Optional[str] = Field(default=None, description="Optional voice identifier.")

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Text cannot be empty or contain only whitespace.")
        return cleaned


class SpeechSynthesizeResponse(BaseModel):
    audio_id: str
    audio_url: str
    file_name: str
    file_size_bytes: int
    rate: int
    volume: float
    voice_id: Optional[str] = None


class VoiceInfo(BaseModel):
    id: str
    name: Optional[str] = None
    languages: Optional[List[Any]] = None
    gender: Optional[str] = None


class SpeechVoicesResponse(BaseModel):
    total: int
    voices: List[VoiceInfo]


# =============================================================================
# EVALUATION (ROUGE) SCHEMAS
# =============================================================================

class EvaluateRequest(BaseModel):
    candidate_summary: str = Field(..., min_length=1, description="System generated summary.")
    reference_summary: str = Field(..., min_length=1, description="Human reference or ground truth summary.")

    @field_validator("candidate_summary", "reference_summary")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Summary cannot be empty or contain only whitespace.")
        return cleaned


class RougeScore(BaseModel):
    precision: float
    recall: float
    f1: float


class EvaluateResponse(BaseModel):
    rouge1: RougeScore
    rouge2: RougeScore
    rougeL: RougeScore


# =============================================================================
# USER AUTHENTICATION SCHEMAS
# =============================================================================

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32, description="Account username.")
    password: str = Field(..., min_length=8, max_length=72, description="Account password (8-72 chars).")

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Username cannot be empty or contain only whitespace.")
        if not all(ch.isalnum() or ch in "-_." for ch in cleaned):
            raise ValueError("Username may only contain letters, digits, '-', '_' and '.'.")
        return cleaned

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v.encode("utf-8")) > 72:
            raise ValueError("Password is too long (max 72 bytes).")
        return v


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, description="Account username.")
    password: str = Field(..., min_length=1, description="Account password.")


class UserResponse(BaseModel):
    id: int
    username: str
    created_at: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_hours: int
    user: UserResponse
