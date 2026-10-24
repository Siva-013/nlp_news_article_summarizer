"""
Vocabulary Router
POST /api/v1/vocabulary/extract
POST /api/v1/vocabulary/detect-new
"""

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_current_user

from api.schemas import (
    VocabularyExtractRequest,
    VocabularyExtractResponse,
    VocabularyDetectNewRequest,
    VocabularyDetectNewResponse,
)
from api.services import run_vocabulary_extraction, run_new_word_detection

router = APIRouter(prefix="/vocabulary", tags=["Vocabulary"])


@router.post(
    "/extract",
    response_model=VocabularyExtractResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract Vocabulary Items",
    description="Extract salient lexical content words and named entities with deterministic heuristic scores (Phase 7).",
)
def extract_vocabulary(
    request: VocabularyExtractRequest,
    user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        result = run_vocabulary_extraction(
            text=request.text,
            article_id=request.article_id or "raw_article",
            max_items=request.max_items or 30,
            include_entities=request.include_entities if request.include_entities is not None else True,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vocabulary extraction failed: {exc}",
        ) from exc


@router.post(
    "/detect-new",
    response_model=VocabularyDetectNewResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect New Words Against Known Vocabulary",
    description="Classifies extracted vocabulary items into 'known_words' and 'new_words' using classical canonical matching (Phase 8).",
)
def detect_new_words(
    request: VocabularyDetectNewRequest,
    user: Dict[str, Any] = Depends(get_current_user),
):
    if not request.text and not request.extraction_record:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'text' or 'extraction_record' must be provided.",
        )

    try:
        result = run_new_word_detection(
            extraction_record=request.extraction_record,
            raw_text=request.text,
            known_vocabulary=request.known_vocabulary,
            max_items=request.max_items or 30,
            include_entities=request.include_entities if request.include_entities is not None else True,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"New-word detection failed: {exc}",
        ) from exc
