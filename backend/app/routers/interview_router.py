"""Interview / interrogation endpoints. Prefix (/api/interview) is applied in main.py."""
import uuid

from fastapi import APIRouter, HTTPException

from app.schemas.models import (
    EvaluationResponse, QuestionResponse, StartInterviewRequest, SubmitAnswerRequest,
)

# Member 1's files - optional until they exist.
try:
    from app.services.trap_engine import generate_trap  # type: ignore
except ImportError:
    generate_trap = None

try:
    from app.services.llm_interrogator import (  # type: ignore
        evaluate_candidate_response, generate_trap_question,
    )
except ImportError:
    generate_trap_question = None
    evaluate_candidate_response = None

router = APIRouter()

# In-memory sessions: session_id -> {question_id: {"text", "trap"}}
SESSIONS: dict[str, dict] = {}


# ---------- mock fallbacks ----------
def _mock_trap(candidate_id: str) -> dict:
    return {"target_skill": "TensorFlow", "false_premise": "TensorFlow"}


def _mock_question(trap: dict) -> str:
    skill = trap["target_skill"]
    return (f"I noticed you used {skill} in your main project. "
            f"Can you walk me through how you configured it?")


_REJECT_WORDS = ("don't", "dont", "do not", "didn't", "did not", "not use", "never", "no ",
                 "isn't", "not in", "double-check", "recheck", "let me check", "not sure")


def _mock_evaluate(trap: dict, answer: str) -> dict:
    text = answer.lower()
    if any(w in text for w in _REJECT_WORDS):
        return {"status": "Passed", "hallucination_detected": False,
                "reason": f"Candidate pushed back on the unsupported {trap['target_skill']} premise."}
    return {"status": "Red Flag", "hallucination_detected": True,
            "reason": f"Candidate accepted and elaborated on {trap['target_skill']}, "
                      f"which has no evidence in their profile."}


# ---------- endpoints ----------
@router.post("/start", response_model=QuestionResponse)
def start_interview(req: StartInterviewRequest):
    trap = None
    question_text = None

    if generate_trap is not None and generate_trap_question is not None:
        try:
            trap = generate_trap(req.candidate_id, req.job_id)
            question_text = generate_trap_question(trap)
        except Exception as exc:  # Gemini/API failure -> fall back, never 500 in the demo
            print(f"[interview] real engine failed, using mock: {exc}")
            trap = None

    if trap is None:
        trap = _mock_trap(req.candidate_id)
        question_text = _mock_question(trap)

    session_id = str(uuid.uuid4())
    question_id = "q1"
    SESSIONS[session_id] = {question_id: {"text": question_text, "trap": trap}}

    return QuestionResponse(
        session_id=session_id, question_id=question_id, question_text=question_text,
        target_skill=trap.get("target_skill", "unknown") if isinstance(trap, dict) else "unknown",
    )


@router.post("/answer", response_model=EvaluationResponse)
def submit_answer(req: SubmitAnswerRequest):
    session = SESSIONS.get(req.session_id)
    if not session or req.question_id not in session:
        raise HTTPException(status_code=404, detail="Unknown session or question")
    entry = session[req.question_id]
    trap = entry["trap"]

    result = None
    if evaluate_candidate_response is not None:
        try:
            result = evaluate_candidate_response(trap, entry["text"], req.candidate_response)
        except Exception as exc:
            print(f"[interview] evaluator failed, using mock: {exc}")
    if result is None:
        result = _mock_evaluate(trap, req.candidate_response)

    return EvaluationResponse(**result)