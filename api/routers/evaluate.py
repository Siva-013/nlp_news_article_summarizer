"""
Evaluation Router
POST /api/v1/evaluate
"""

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_current_user

from api.schemas import EvaluateRequest, EvaluateResponse
from api.services import run_rouge_evaluation

router = APIRouter(prefix="/evaluate", tags=["Evaluation"])


@router.post(
    "",
    response_model=EvaluateResponse,
    status_code=status.HTTP_200_OK,
    summary="Compute ROUGE Evaluation Metrics",
    description="Evaluate precision, recall, and F1 scores for ROUGE-1, ROUGE-2, and ROUGE-L between candidate and reference summaries (Phase 6).",
)
def evaluate_summary(
    request: EvaluateRequest,
    user: Dict[str, Any] = Depends(get_current_user),
):
    try:
        result = run_rouge_evaluation(
            candidate_summary=request.candidate_summary,
            reference_summary=request.reference_summary,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation calculation error: {exc}",
        ) from exc
