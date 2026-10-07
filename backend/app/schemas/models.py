"""Pydantic data contracts shared by the dashboard and interview routers."""
from typing import Literal

from pydantic import BaseModel, Field


# ---------- Dashboard ----------
class CandidateSuitability(BaseModel):
    candidate_id: str
    name: str
    suitability_score: float = Field(..., ge=0, le=100)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)


class SkillStatus(BaseModel):
    skill: str
    status: Literal["verified_strength", "skill_gap", "unverified"]
    market_demand: int = Field(0, ge=0, le=100, description="Relative market demand, 0-100")


class SkillGapResponse(BaseModel):
    """Flat market-vs-candidate table (replaces the 3D graph)."""
    candidate_id: str
    target_role: str
    skills: list[SkillStatus]


# ---------- Interviews ----------
class StartInterviewRequest(BaseModel):
    candidate_id: str
    interview_type: Literal["technical", "behavioral"] = "technical"


class InterviewQuestion(BaseModel):
    question_id: int
    question_type: Literal[
        "true_premise",
        "false_premise_trap",
        "open_ended",
        "behavioral_pressure",
    ]
    prompt_text: str
    is_trap: bool
    targeted_trait: str


class GeneratedInterviewResponse(BaseModel):
    candidate_id: str
    strong_trait: str
    weak_trait: str
    questions: list[InterviewQuestion]


class InterviewStartResponse(BaseModel):
    status: Literal["success"]
    data: GeneratedInterviewResponse


class SubmitAnswerRequest(BaseModel):
    original_question: str
    candidate_answer: str
    attempt_number: Literal[1, 2]


class EvaluationResponse(BaseModel):
    status: Literal["VERIFIED", "FLAGGED", "FOLLOW_UP", "HUMAN_REVIEW"]
    reasoning: str
    next_ai_reply: str


class InterviewAnswerResponse(BaseModel):
    status: Literal["success"]
    evaluation: EvaluationResponse


class InterviewCandidatesResponse(BaseModel):
    interview_type: Literal["technical", "behavioral"]
    candidates: list[str]