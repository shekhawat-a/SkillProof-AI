"""Interview / interrogation endpoints. Prefix (/api/interview) is applied in main.py."""

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from app.schemas.models import (
    InterviewAnswerResponse,
    InterviewCandidatesResponse,
    InterviewStartResponse,
    StartInterviewRequest,
    SubmitAnswerRequest,
)
from app.services.evaluation_engine import evaluate_trap_answer
from app.services.trap_engine import (
    JDS_FILE_PATH,
    SDS_FILE_PATH,
    generate_behavioral_trap,
    generate_technical_trap,
    list_candidate_ids,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/candidates", response_model=InterviewCandidatesResponse)
def get_interview_candidates(
    interview_type: Literal["technical", "behavioral"] = Query(default="technical"),
) -> InterviewCandidatesResponse:
    """List IDs from the dataset used by the selected interview type."""
    data_path = JDS_FILE_PATH if interview_type == "technical" else SDS_FILE_PATH
    try:
        candidate_ids = list_candidate_ids(data_path)
    except Exception as exc:
        logger.exception("Unable to load %s interview candidates", interview_type)
        raise HTTPException(
            status_code=500,
            detail=f"Unable to load {interview_type} interview candidates",
        ) from exc

    return InterviewCandidatesResponse(
        interview_type=interview_type,
        candidates=candidate_ids,
    )


@router.post("/start", response_model=InterviewStartResponse)
def start_interview(req: StartInterviewRequest) -> InterviewStartResponse:
    """
    Generates a 3-question dynamic interview (True Premise, False Premise Trap, Open Ended).
    """
    generator = (
        generate_behavioral_trap
        if req.interview_type == "behavioral"
        else generate_technical_trap
    )
    data = generator(req.candidate_id)
    if "error" in data:
        status_code = {
            "candidate_not_found": 404,
            "dataset_error": 500,
            "upstream_error": 502,
        }.get(data.get("error_code"), 502)
        raise HTTPException(status_code=status_code, detail=data["error"])

    return InterviewStartResponse(status="success", data=data)


@router.post("/answer", response_model=InterviewAnswerResponse)
def submit_answer(req: SubmitAnswerRequest) -> InterviewAnswerResponse:
    """
    Evaluates candidate's answer using the strict 'Double-Down' psychological protocol.
    Returns: VERIFIED, FLAGGED, FOLLOW_UP, or HUMAN_REVIEW.
    """
    evaluation_result = evaluate_trap_answer(
        original_question=req.original_question,
        candidate_answer=req.candidate_answer,
        attempt_number=req.attempt_number,
    )
    if "error" in evaluation_result:
        raise HTTPException(status_code=502, detail=evaluation_result["error"])

    return InterviewAnswerResponse(status="success", evaluation=evaluation_result)