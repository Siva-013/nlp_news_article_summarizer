"""
Vocabulary Notes Router
CRUD operations for learner notes persistence (Phase 9)
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from typing import Any, Dict

from api.dependencies import get_db, get_current_user
from api.schemas import (
    NoteCreateRequest,
    NoteUpdateRequest,
    NoteResponse,
    NoteListResponse,
)
from database.notes_history import NotesHistoryManager

router = APIRouter(prefix="/notes", tags=["Vocabulary Notes"])


@router.post(
    "",
    response_model=NoteResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Vocabulary Note",
    description="Save a new vocabulary study note or example sentence, optionally linking to article history.",
)
def create_vocabulary_note(
    request: NoteCreateRequest,
    db: NotesHistoryManager = Depends(get_db),
    user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        new_id = db.create_note(
            content=request.content,
            term=request.term,
            example_sentence=request.example_sentence,
            article_history_id=request.article_history_id,
            user_id=user["id"],
        )
        record = db.get_note(new_id, user_id=user["id"])
        if not record:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve newly created note.",
            )
        return record
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while creating note: {exc}",
        ) from exc


@router.get(
    "",
    response_model=NoteListResponse,
    status_code=status.HTTP_200_OK,
    summary="List Vocabulary Notes",
    description="List learner vocabulary notes with pagination and optional filtering by term or article history ID.",
)
def list_vocabulary_notes(
    limit: int = Query(20, ge=1, le=200, description="Page limit"),
    offset: int = Query(0, ge=0, description="Page offset"),
    term: Optional[str] = Query(None, description="Filter by exact term"),
    article_history_id: Optional[int] = Query(None, description="Filter by article history ID"),
    db: NotesHistoryManager = Depends(get_db),
    user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        total = db.count_notes(term=term, article_history_id=article_history_id, user_id=user["id"])
        items = db.list_notes(
            limit=limit,
            offset=offset,
            term=term,
            article_history_id=article_history_id,
            user_id=user["id"],
        )
        return {"total": total, "limit": limit, "offset": offset, "items": items}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while querying notes: {exc}",
        ) from exc


@router.get(
    "/search",
    response_model=NoteListResponse,
    status_code=status.HTTP_200_OK,
    summary="Search Vocabulary Notes",
    description="Search notes across content, term, and example sentence.",
)
def search_vocabulary_notes(
    q: str = Query(..., min_length=1, description="Search keyword"),
    limit: int = Query(20, ge=1, le=200, description="Page limit"),
    offset: int = Query(0, ge=0, description="Page offset"),
    db: NotesHistoryManager = Depends(get_db),
    user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        total = db.count_search_notes(query=q, user_id=user["id"])
        items = db.search_notes(query=q, limit=limit, offset=offset, user_id=user["id"])
        return {"total": total, "limit": limit, "offset": offset, "items": items}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while searching notes: {exc}",
        ) from exc


@router.get(
    "/{note_id}",
    response_model=NoteResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Vocabulary Note by ID",
    description="Retrieve a single vocabulary note by unique ID.",
)
def get_vocabulary_note(
    note_id: int,
    db: NotesHistoryManager = Depends(get_db),
    user: Dict[str, Any] = Depends(get_current_user),
):
    record = db.get_note(note_id, user_id=user["id"])
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vocabulary note with ID {note_id} not found.",
        )
    return record


@router.put(
    "/{note_id}",
    response_model=NoteResponse,
    status_code=status.HTTP_200_OK,
    summary="Update Vocabulary Note",
    description="Update content, term, example sentence, or article association of an existing note.",
)
def update_vocabulary_note(
    note_id: int,
    request: NoteUpdateRequest,
    db: NotesHistoryManager = Depends(get_db),
    user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        updated = db.update_note(
            note_id=note_id,
            content=request.content,
            term=request.term,
            example_sentence=request.example_sentence,
            article_history_id=request.article_history_id,
            clear_term=request.clear_term,
            clear_example=request.clear_example,
            clear_article_history=request.clear_article_history,
            user_id=user["id"],
        )
        if not updated:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Vocabulary note with ID {note_id} not found.",
            )
        return db.get_note(note_id, user_id=user["id"])
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while updating note: {exc}",
        ) from exc


@router.delete(
    "/{note_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete Vocabulary Note",
    description="Delete a vocabulary note by unique ID.",
)
def delete_vocabulary_note(
    note_id: int,
    db: NotesHistoryManager = Depends(get_db),
    user: Dict[str, Any] = Depends(get_current_user),
):
    deleted = db.delete_note(note_id, user_id=user["id"])
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vocabulary note with ID {note_id} not found.",
        )
    return {"status": "success", "deleted_id": note_id, "message": "Vocabulary note deleted successfully."}
