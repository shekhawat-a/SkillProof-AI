"""SkillProof AI trap engine (Groq + Dual CSV Edition).

Dynamically reads either the JDS (Technical) or SDS (Personality) datasets.
Generates targeted Technical or Behavioral trap questions using Groq (Llama 3).
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
from pydantic import BaseModel, ValidationError

load_dotenv()

logger = logging.getLogger(__name__)

# Groq models verified to support chat completions and strict JSON output.
GROQ_MODELS = (
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
)

# File Paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
JDS_FILE_PATH = DATA_DIR / "JDS Skill Traits.xlsx"
SDS_FILE_PATH = DATA_DIR / "SDS Personality Traits.xlsx"

ID_COLUMN_CANDIDATES = ("id", "candidate_id", "candidateid", "s_no", "reference_no")

# Ignore these columns when calculating highest/lowest traits
NON_FEATURE_COLUMNS = {
    "id", "candidate_id", "candidateid", "s_no", "reference_no",
    "salary_hike_high_or_low", "success_ classification_ high_low"
}

_client: Groq | None = None

# ==========================================
# PYDANTIC SCHEMAS
# ==========================================
class Question(BaseModel):
    question_id: int
    question_type: Literal["true_premise", "false_premise_trap", "open_ended", "behavioral_pressure"]
    prompt_text: str
    is_trap: bool
    targeted_trait: str

class InterviewOutput(BaseModel):
    candidate_id: str
    strong_trait: str
    weak_trait: str
    questions: List[Question]

# ==========================================
# CORE LOGIC
# ==========================================

def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set in .env")
        _client = Groq(api_key=api_key)
    return _client

def _resolve_id_column(columns: pd.Index) -> str | None:
    normalized = {str(col).strip().lower(): col for col in columns}
    for name in ID_COLUMN_CANDIDATES:
        if name in normalized:
            return str(normalized[name])
    return None

def load_candidate_profile(candidate_id: str, data_path: Path) -> tuple[str, str]:
    """Loads an Excel file, finds the candidate, and returns their highest and lowest scoring traits."""
    candidate_id = str(candidate_id).strip()
    
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    # FIX: Use read_excel instead of read_csv since your files are .xlsx
    try:
        df = pd.read_excel(data_path, engine="openpyxl")
    except ImportError:
        raise ImportError("Please install openpyxl by running: pip install openpyxl")
    except Exception as exc:
        raise ValueError(f"Failed to read Excel file {data_path}: {exc}")

    id_column = _resolve_id_column(df.columns)
    
    if id_column is None:
        raise ValueError(f"Missing ID column in {data_path.name}")

    numeric_columns = [col for col in df.select_dtypes(include="number").columns]
    normalized_ids = df[id_column].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    matches = df[normalized_ids == candidate_id]
    
    if matches.empty:
        raise ValueError(f"Candidate '{candidate_id}' not found in {data_path.name}")

    candidate_row = matches.iloc[0]
    
    traits: dict[str, float] = {}
    for col in numeric_columns:
        if str(col).strip().lower() in NON_FEATURE_COLUMNS:
            continue
        if not pd.isna(candidate_row[col]):
            traits[str(col)] = float(candidate_row[col])

    if not traits:
        raise ValueError(f"No usable data for candidate '{candidate_id}'.")

    # Clean up names for the LLM (e.g., "ai_and_ml_skills" -> "Ai And Ml Skills")
    strong_raw = max(traits, key=traits.get)
    weak_raw = min(traits, key=traits.get)
    
    strong_trait = strong_raw.replace("_", " ").title()
    weak_trait = weak_raw.replace("_", " ").title()
    
    return strong_trait, weak_trait


def _call_groq(system_prompt: str, user_prompt: str) -> InterviewOutput:
    client = _get_client()
    for model_index, model in enumerate(GROQ_MODELS):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt.strip()},
                    {"role": "user", "content": user_prompt.strip()}
                ],
                response_format={"type": "json_object"},
                temperature=0.2, 
            )
            raw_text = response.choices[0].message.content
            return InterviewOutput.model_validate_json(raw_text)
        except Exception as exc:
            logger.warning(f"Groq model {model} failed: {exc}")
            if model_index == len(GROQ_MODELS) - 1:
                raise 

# ==========================================
# INTERVIEW GENERATORS
# ==========================================

def generate_technical_trap(candidate_id: str) -> dict[str, Any]:
    """Generates a Technical Skills interview using the JDS dataset."""
    try:
        strong, weak = load_candidate_profile(candidate_id, JDS_FILE_PATH)
    except Exception as exc:
        return {"error": str(exc)}

    schema_str = json.dumps(InterviewOutput.model_json_schema(), indent=2)
    system_prompt = f"You are a strict Technical Recruiter. Output ONLY valid JSON matching this schema:\n{schema_str}"
    
    user_prompt = f"""
    Candidate id: {candidate_id}
    Strongest Technical Skill: {strong}
    Weakest Technical Skill: {weak}

    Produce exactly 3 questions:
    1. question_type "true_premise": validate real depth in {strong}. is_trap = false.
    2. question_type "false_premise_trap": treat {weak} as if they claimed to be an expert in it. Ask them to explain a highly complex, fake architecture they built using it to see if they hallucinate. is_trap = true.
    3. question_type "open_ended": A messy real-world data engineering problem. is_trap = false.
    """
    
    try:
        return _call_groq(system_prompt, user_prompt).model_dump()
    except Exception as exc:
        return {"error": f"Failed to generate technical trap: {exc}"}


def generate_behavioral_trap(candidate_id: str) -> dict[str, Any]:
    """Generates a Personality/Behavioral interview using the SDS dataset."""
    try:
        strong, weak = load_candidate_profile(candidate_id, SDS_FILE_PATH)
    except Exception as exc:
        return {"error": str(exc)}

    schema_str = json.dumps(InterviewOutput.model_json_schema(), indent=2)
    system_prompt = f"You are an elite Behavioral Psychologist/HR Recruiter. Output ONLY valid JSON matching this schema:\n{schema_str}"
    
    user_prompt = f"""
    Candidate id: {candidate_id}
    Dominant Personality Trait: {strong}
    Weakest Personality Trait: {weak}

    Produce exactly 3 questions:
    1. question_type "true_premise": Validate their {strong} trait with a situational workplace question. is_trap = false.
    2. question_type "behavioral_pressure": Put them in a highly stressful hypothetical scenario that specifically targets their lack of {weak}. is_trap = true.
    3. question_type "open_ended": Ask how they handle severe conflict with a co-worker. is_trap = false.
    """
    
    try:
        return _call_groq(system_prompt, user_prompt).model_dump()
    except Exception as exc:
        return {"error": f"Failed to generate behavioral trap: {exc}"}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n--- GENERATING TECHNICAL INTERVIEW (JDS DATASET) ---")
    tech_result = generate_technical_trap("2809")
    print(json.dumps(tech_result, indent=2))

    print("\n--- GENERATING BEHAVIORAL INTERVIEW (SDS DATASET) ---")
    # Using ID 8120 from your SDS dataset
    behav_result = generate_behavioral_trap("8120") 
    print(json.dumps(behav_result, indent=2))