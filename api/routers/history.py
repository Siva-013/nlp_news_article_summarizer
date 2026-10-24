"""
Article History Router
CRUD operations for article history persistence (Phase 9)
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from typing import Any, Dict, Optional

from api.dependencies import get_db, get_current_user
from api.schemas import (
    ArticleHistoryCreateRequest,
    ArticleHistoryResponse,
    ArticleHistoryListResponse,
)
from database.notes_history import NotesHistoryManager

router = APIRouter(prefix="/history", tags=["Article History"])


@router.post(
    "",
    response_model=ArticleHistoryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Save Article History Entry",
    description="Save an extractive summary and vocabulary metadata to SQLite article history.",
)
def create_article_history_entry(
    request: ArticleHistoryCreateRequest,
    db: NotesHistoryManager = Depends(get_db),
    user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        new_id = db.save_article_history(
            summary=request.summary,
            method=request.method,
            title=request.title,
            article_text=request.article_text,
            source_article_id=request.source_article_id,
            extracted_vocabulary=request.extracted_vocabulary,
            new_vocabulary=request.new_vocabulary,
            metadata=request.metadata,
            user_id=user["id"],
        )
        record = db.get_article_history(new_id, user_id=user["id"])
        if not record:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve newly created history record.",
            )
        return record
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while saving history: {exc}",
        ) from exc


@router.get(
    "",
    response_model=ArticleHistoryListResponse,
    status_code=status.HTTP_200_OK,
    summary="List Article History",
    description="List historical summarized articles with pagination and method filtering.",
)
def list_article_history(
    limit: int = Query(50, ge=1, le=500, description="Page limit"),
    offset: int = Query(0, ge=0, description="Page offset"),
    method: Optional[str] = Query(None, description="Filter by summarization method"),
    db: NotesHistoryManager = Depends(get_db),
    user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        total = db.count_article_history(method=method, user_id=user["id"])
        items = db.list_article_history(limit=limit, offset=offset, method=method, user_id=user["id"])
        return {"total": total, "limit": limit, "offset": offset, "items": items}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while querying history: {exc}",
        ) from exc


@router.get(
    "/search",
    response_model=ArticleHistoryListResponse,
    status_code=status.HTTP_200_OK,
    summary="Search Article History",
    description="Search article history by title, text, summary, or source ID.",
)
def search_article_history(
    q: str = Query(..., min_length=1, description="Search keyword"),
    limit: int = Query(50, ge=1, le=500, description="Page limit"),
    offset: int = Query(0, ge=0, description="Page offset"),
    db: NotesHistoryManager = Depends(get_db),
    user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        total = db.count_search_article_history(query=q, user_id=user["id"])
        items = db.search_article_history(query=q, limit=limit, offset=offset, user_id=user["id"])
        return {"total": total, "limit": limit, "offset": offset, "items": items}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while searching history: {exc}",
        ) from exc


@router.get(
    "/{history_id}",
    response_model=ArticleHistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Article History by ID",
    description="Retrieve full historical record including extracted vocabulary and metadata.",
)
def get_article_history(
    history_id: int,
    db: NotesHistoryManager = Depends(get_db),
    user: Dict[str, Any] = Depends(get_current_user),
):
    record = db.get_article_history(history_id, user_id=user["id"])
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Article history record with ID {history_id} not found.",
        )
    return record


@router.delete(
    "/{history_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete Article History by ID",
    description="Delete an article history entry. Associated notes have their foreign key set to NULL (ON DELETE SET NULL).",
)
def delete_article_history(
    history_id: int,
    db: NotesHistoryManager = Depends(get_db),
    user: Dict[str, Any] = Depends(get_current_user),
):
    deleted = db.delete_article_history(history_id, user_id=user["id"])
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Article history record with ID {history_id} not found.",
        )
    return {"status": "success", "deleted_id": history_id, "message": "Article history entry deleted successfully."}
