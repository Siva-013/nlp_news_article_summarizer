"""
Summarization Router
POST /api/v1/summarize
"""

from fastapi import APIRouter, HTTPException, status

from api.schemas import SummarizeRequest, SummarizeResponse
from api.services import run_summarization

router = APIRouter(prefix="/summarize", tags=["Summarization"])


@router.post(
    "",
    response_model=SummarizeResponse,
    status_code=status.HTTP_200_OK,
    summary="Extractive News Summarization",
    description="Generate an extractive summary of a news article using Frequency, TF-IDF, or TextRank algorithms.",
)
def summarize_article(request: SummarizeRequest):
    try:
        result = run_summarization(
            text=request.text,
            method=request.method,
            num_sentences=request.num_sentences,
            article_id=request.article_id or "custom_article",
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summarization processing error: {exc}",
        ) from exc
