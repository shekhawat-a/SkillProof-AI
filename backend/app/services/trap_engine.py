"""SkillProof AI trap engine (Groq Edition).

Reads a candidate's skill scores from the JDS Excel dataset, identifies the
strongest and weakest skills, and asks Groq (Llama 3) to return a strict JSON 
interview payload matching ``InterviewOutput``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, List, Literal

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, ConfigDict, ValidationError

load_dotenv()

logger = logging.getLogger(__name__)

# Groq models supporting strict structured outputs (primary, then fallbacks).
GROQ_MODELS = ("openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b")

DATA_FILENAME = "JDS Skill Traits.xlsx"
EXCEL_PATH = Path(__file__).resolve().parent.parent / "data" / DATA_FILENAME

ID_COLUMN_CANDIDATES = ("id", "candidate_id", "candidateid")
NON_SKILL_COLUMNS = {
    "id",
    "candidate_id",
    "candidateid",
    "salary_hike_high_or_low",
}

_client: Groq | None = None

# ==========================================
# PYDANTIC SCHEMAS (Defined inline for safety)
# ==========================================
class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: int
    question_type: Literal["true_premise", "false_premise_trap", "open_ended"]
    prompt_text: str
    is_trap: bool
    targeted_skill: str

class InterviewOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    strong_skill: str
    weak_skill: str
    questions: List[Question]

# ==========================================
# CORE LOGIC
# ==========================================

def _get_client() -> Groq:
    """Return a cached Groq client, configuring it from ``GROQ_API_KEY``."""
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file before generating traps."
            )
        _client = Groq(api_key=api_key)
    return _client


def _resolve_id_column(columns: pd.Index) -> str | None:
    """Find a supported candidate identifier column, ignoring case and spaces."""
    normalized = {str(col).strip().lower(): col for col in columns}
    for name in ID_COLUMN_CANDIDATES:
        if name in normalized:
            return str(normalized[name])
    return None


def _extract_skill_scores(row: pd.Series, numeric_columns: list[str]) -> dict[str, float]:
    """Return numeric skill scores, skipping id / outcome columns."""
    skills: dict[str, float] = {}
    for col in numeric_columns:
        if str(col).strip().lower() in NON_SKILL_COLUMNS:
            continue
        value = row[col]
        if pd.isna(value):
            continue
        skills[str(col)] = float(value)
    return skills


def load_candidate_skills(candidate_id: str, excel_path: Path = EXCEL_PATH) -> tuple[str, str]:
    """Load the Excel dataset and return ``(strong_skill, weak_skill)`` for a candidate."""
    candidate_id = str(candidate_id).strip()
    if not candidate_id:
        raise ValueError("candidate_id must not be empty.")

    if not excel_path.is_file():
        raise FileNotFoundError(
            f"Skill dataset not found at {excel_path}. "
            "Expected an Excel file named 'JDS Skill Traits.xlsx'."
        )

    try:
        df = pd.read_excel(excel_path, engine="openpyxl")
    except ImportError as exc:
        raise ImportError("Reading .xlsx files requires openpyxl. Install it with: pip install openpyxl") from exc
    except Exception as exc:
        raise ValueError(f"Failed to read Excel workbook {excel_path}: {exc}") from exc

    if df.empty:
        raise ValueError(f"Excel workbook {excel_path.name} has no rows.")

    id_column = _resolve_id_column(df.columns)
    if id_column is None:
        raise ValueError(f"Missing an id column. Looked for {list(ID_COLUMN_CANDIDATES)}.")

    numeric_columns = [col for col in df.select_dtypes(include="number").columns]
    if not numeric_columns:
        raise ValueError("No numeric skill columns found in Excel.")

    # FIX: Handle cases where pandas reads IDs as floats (e.g., 2809.0 instead of "2809")
    normalized_ids = df[id_column].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    matches = df[normalized_ids == candidate_id]
    
    if matches.empty:
        raise ValueError(f"Candidate '{candidate_id}' was not found in column '{id_column}'.")

    candidate_row = matches.iloc[0]
    skills = _extract_skill_scores(candidate_row, numeric_columns)
    if not skills:
        raise ValueError(f"No usable skill scores for candidate '{candidate_id}'.")

    strong_skill = max(skills, key=skills.get)
    weak_skill = min(skills, key=skills.get)
    return strong_skill, weak_skill


def _call_groq(candidate_id: str, strong_skill: str, weak_skill: str) -> InterviewOutput:
    """Call Groq API with structured JSON output constrained to InterviewOutput."""
    client = _get_client()
    
    # Inject Pydantic schema into the system prompt to guarantee structure
    schema_str = json.dumps(InterviewOutput.model_json_schema(), indent=2)
    
    system_prompt = f"""
You are a strict Data Science technical recruiter generating a SkillProof AI interview.
You MUST output ONLY valid JSON. Do not include markdown formatting like ```json.
The JSON must strictly adhere to the following JSON Schema:
{schema_str}
"""

    user_prompt = f"""
Candidate id: {candidate_id}
Strongest skill (from scored traits): {strong_skill}
Weakest skill (from scored traits): {weak_skill}

Produce exactly 3 questions in the JSON array:
1. question_type "true_premise": validate real depth in {strong_skill}. is_trap = false. targeted_skill = {strong_skill}.
2. question_type "false_premise_trap": treat {weak_skill} as if it appeared as a strong resume claim and ask them to explain a complex production architecture they built with it. is_trap = true. targeted_skill = {weak_skill}.
3. question_type "open_ended": messy real-world data / evaluation judgment. is_trap = false. targeted_skill can be a data-quality skill.

Rules:
- candidate_id, strong_skill, and weak_skill must match the values above exactly.
- question_id must be 1, 2, and 3 in that order.
- prompt_text must be a specific technical interview question, not a description of the question.
"""

    for model_index, model in enumerate(GROQ_MODELS):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt.strip()},
                    {"role": "user", "content": user_prompt.strip()}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "interview_output",
                        "strict": True,
                        "schema": InterviewOutput.model_json_schema(),
                    },
                },
                temperature=0.2, # Low temperature for deterministic output
            )
            
            raw_text = response.choices[0].message.content
            if not raw_text:
                raise RuntimeError("Groq returned an empty response.")
                
            return InterviewOutput.model_validate_json(raw_text)
            
        except Exception as exc:
            logger.warning(f"Groq model {model} failed: {exc}")
            if model_index == len(GROQ_MODELS) - 1:
                raise # Re-raise if all fallback models fail


def generate_ds_trap(candidate_id: str) -> dict[str, Any]:
    """Generate a three-question false-premise trap interview for one candidate."""
    candidate_id = str(candidate_id).strip()
    if not candidate_id:
        return {"error": "candidate_id must not be empty."}

    logger.info("Analyzing candidate %s", candidate_id)

    try:
        strong_skill, weak_skill = load_candidate_skills(candidate_id)
    except Exception as exc:
        logger.exception("Skill dataset error")
        return {"error": str(exc)}

    logger.info("Candidate %s strongest skill: %s", candidate_id, strong_skill)
    logger.info("Candidate %s weakest skill: %s", candidate_id, weak_skill)

    try:
        interview = _call_groq(candidate_id, strong_skill, weak_skill)
        
        # Validation checks
        if (interview.candidate_id != str(candidate_id) or 
            interview.strong_skill != strong_skill or 
            interview.weak_skill != weak_skill):
            raise ValueError("Groq response did not preserve the candidate and skill values.")

        expected_types = ["true_premise", "false_premise_trap", "open_ended"]
        if [q.question_type for q in interview.questions] != expected_types:
            raise ValueError("Groq response questions were not returned in the required order.")
            
        if [q.question_id for q in interview.questions] != [1, 2, 3]:
            raise ValueError("Groq response question IDs must be 1, 2, and 3 in order.")

        return interview.model_dump()
        
    except ValidationError as exc:
        logger.exception("Groq JSON did not match InterviewOutput")
        return {"error": f"Groq response failed schema validation: {exc}"}
    except Exception as exc:
        logger.exception("Groq API call failed")
        return {"error": f"Failed to generate trap: {exc}"}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Ensure you have GROQ_API_KEY in your environment or .env file
    result = generate_ds_trap("2809")
    print(json.dumps(result, indent=2))