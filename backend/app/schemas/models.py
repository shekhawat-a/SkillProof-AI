"""Pydantic data contracts shared by the dashboard and interview routers."""
from typing import Literal

from pydantic import BaseModel, Field


# ---------- Dashboard ----------
class CandidateSuitability(BaseModel):
    candidate_id: str
    name: str
    suitability_score: int = Field(..., ge=1, le=100)
    strengths: list[str] = []
    weaknesses: list[str] = []


class SkillStatus(BaseModel):
    skill: str
    status: Literal["verified_strength", "skill_gap", "unverified"]
    market_demand: int = Field(0, ge=0, le=100, description="Relative market demand, 0-100")


class SkillGapResponse(BaseModel):
    """Flat market-vs-candidate table (replaces the 3D graph)."""
    candidate_id: str
    target_role: str
    skills: list[SkillStatus]


# ---------- Interview / Chat ----------
class StartInterviewRequest(BaseModel):
    candidate_id: str
    job_id: str


class QuestionResponse(BaseModel):
    session_id: str
    question_id: str
    question_text: str
    target_skill: str


class SubmitAnswerRequest(BaseModel):
    session_id: str
    question_id: str
    candidate_response: str


class EvaluationResponse(BaseModel):
    status: Literal["Passed", "Red Flag"]
    reason: str
    hallucination_detected: bool