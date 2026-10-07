"""Recruiter dashboard endpoints. Prefix (/api/dashboard) is applied in main.py."""
import logging
import math
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from app.schemas.models import CandidateSuitability, SkillGapResponse, SkillStatus

router = APIRouter()
logger = logging.getLogger(__name__)

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "JDS Skill Traits.xlsx"
SKILL_COLUMNS = (
    "big_data_skills",
    "maths-stats_skills",
    "coding_skills",
    "ai_and_ml_skills",
    "dashboard_and_storytelling_skills",
)
STRENGTH_THRESHOLD = 3.5


def _load_skill_data() -> pd.DataFrame:
    """Load candidate skill scores from the JDS workbook."""
    try:
        data = pd.read_excel(DATA_FILE, sheet_name="JDS")
    except Exception as exc:
        logger.exception("Unable to load dashboard dataset: %s", DATA_FILE)
        raise HTTPException(
            status_code=500,
            detail="Unable to load candidate dataset",
        ) from exc

    required_columns = {"id", *SKILL_COLUMNS}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        logger.error("Dashboard dataset is missing columns: %s", sorted(missing_columns))
        raise HTTPException(
            status_code=500,
            detail="Candidate dataset has an invalid format",
        )
    return data


def _candidate_id(value: object) -> str:
    if pd.isna(value):
        raise HTTPException(status_code=500, detail="Candidate dataset contains an empty id")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _skill_scores(row: pd.Series) -> dict[str, float]:
    scores: dict[str, float] = {}
    for column in SKILL_COLUMNS:
        try:
            score = float(row[column])
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=500,
                detail="Candidate dataset contains an invalid skill score",
            ) from exc
        if not math.isfinite(score):
            raise HTTPException(
                status_code=500,
                detail="Candidate dataset contains an invalid skill score",
            )
        scores[column] = score
    return scores


def _to_candidate(row: pd.Series) -> CandidateSuitability:
    scores = _skill_scores(row)
    strengths = [
        column.replace("_", " ").replace("-", " ").title()
        for column, score in scores.items()
        if score >= STRENGTH_THRESHOLD
    ]
    weaknesses = [
        column.replace("_", " ").replace("-", " ").title()
        for column, score in scores.items()
        if score < STRENGTH_THRESHOLD
    ]
    suitability_score = round(sum(scores.values()) / len(scores) * 20)

    return CandidateSuitability(
        candidate_id=_candidate_id(row["id"]),
        name=f"Candidate {_candidate_id(row['id'])}",
        suitability_score=min(100, max(1, suitability_score)),
        strengths=strengths,
        weaknesses=weaknesses,
    )


def _find_row(candidate_id: str) -> pd.Series:
    data = _load_skill_data()
    for _, row in data.iterrows():
        if _candidate_id(row["id"]) == candidate_id:
            return row
    raise HTTPException(status_code=404, detail=f"Candidate '{candidate_id}' not found")


@router.get("/candidates", response_model=list[CandidateSuitability])
def list_candidates() -> list[CandidateSuitability]:
    return [_to_candidate(row) for _, row in _load_skill_data().iterrows()]


@router.get("/candidate/{candidate_id}", response_model=CandidateSuitability)
def get_candidate(candidate_id: str) -> CandidateSuitability:
    return _to_candidate(_find_row(candidate_id))


@router.get("/gap/{candidate_id}", response_model=SkillGapResponse)
def get_skill_gap(candidate_id: str) -> SkillGapResponse:
    row = _find_row(candidate_id)
    scores = _skill_scores(row)
    skills = [
        SkillStatus(
            skill=column.replace("_", " ").replace("-", " ").title(),
            status="verified_strength" if score >= STRENGTH_THRESHOLD else "skill_gap",
        )
        for column, score in scores.items()
    ]
    return SkillGapResponse(
        candidate_id=_candidate_id(row["id"]),
        target_role="Data Scientist",
        skills=skills,
    )
