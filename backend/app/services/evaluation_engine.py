"""SkillProof AI Evaluation Engine.

Evaluates candidate responses to false-premise traps using a strict
"Double-Down" psychological protocol to detect resume padding.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Literal

from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel

load_dotenv()
logger = logging.getLogger(__name__)

# ==========================================
# PYDANTIC SCHEMAS
# ==========================================
class EvaluationResponse(BaseModel):
    status: Literal["VERIFIED", "FLAGGED", "FOLLOW_UP", "HUMAN_REVIEW"]
    reasoning: str
    next_ai_reply: str

_client: Groq | None = None

def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set in .env")
        _client = Groq(api_key=api_key)
    return _client

# ==========================================
# CORE LOGIC: THE DOUBLE-DOWN PROTOCOL
# ==========================================
def evaluate_trap_answer(original_question: str, candidate_answer: str, attempt_number: int) -> dict:
    """
    Evaluates the candidate's answer to a false premise trap.
    Returns a dictionary matching the EvaluationResponse schema.
    """
    client = _get_client()
    schema_str = json.dumps(EvaluationResponse.model_json_schema(), indent=2)
    
    system_prompt = f"""
    You are an elite HR-tech behavioral lie-detector AI. 
    Your job is to evaluate a candidate's response to a "false premise trap" (where the AI deliberately claims the candidate built a fake, highly complex project).

    EVALUATION RULES (The Double-Down Strategy):
    1. HONEST (Corrects AI): If the candidate immediately denies the fake premise (e.g., "I didn't build that"), set status to 'VERIFIED'. next_ai_reply should be a positive transition.
    
    2. LIAR - ATTEMPT 1 (Agrees/Hallucinates): If attempt_number == 1 and they agree/invent details, DO NOT warn them. Act impressed. "Double-Down" by asking a highly technical, even harder follow-up about their fake project to let them dig their own grave. Set status to 'FOLLOW_UP'.
    
    3. LIAR - ATTEMPT 2 (The Follow-Up Evaluation): If attempt_number == 2:
       - If they confidently continue hallucinating fake details -> set status to 'FLAGGED' (Auto-Reject).
       - If they get scared, hedge, or backpedal (e.g., "Actually, my senior did it") -> set status to 'HUMAN_REVIEW'.
       In Attempt 2, next_ai_reply should just be a neutral closing like "Thank you, moving on to the next topic."

    Output STRICTLY as a JSON object matching this schema:
    {schema_str}
    """

    user_prompt = f"""
    Attempt Number: {attempt_number}
    Original Trap Question: {original_question}
    Candidate's Answer: {candidate_answer}
    """

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,  # Keep it low for deterministic evaluation
        )
        raw_text = response.choices[0].message.content
        return EvaluationResponse.model_validate_json(raw_text).model_dump()
    except Exception as exc:
        logger.error(f"Groq Evaluation failed: {exc}")
        return {"error": f"Failed to evaluate answer: {exc}"}

# ==========================================
# LOCAL TESTING
# ==========================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n--- TESTING SCENARIO 1: HONEST CANDIDATE ---")
    res1 = evaluate_trap_answer(
        original_question="I saw you built a custom C++ Garbage Collector. Walk me through it.",
        candidate_answer="I think there is a misunderstanding. I never built a custom Garbage Collector in C++.",
        attempt_number=1
    )
    print(json.dumps(res1, indent=2))

    print("\n--- TESTING SCENARIO 2: LIAR (ATTEMPT 1) ---")
    res2 = evaluate_trap_answer(
        original_question="I saw you built a custom C++ Garbage Collector. Walk me through it.",
        candidate_answer="Yes, I built it using smart pointers and memory pools to optimize real-time latency.",
        attempt_number=1
    )
    print(json.dumps(res2, indent=2))