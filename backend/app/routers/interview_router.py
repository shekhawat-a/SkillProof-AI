"""Interview / interrogation endpoints. Prefix (/api/interview) is applied in main.py."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Naye AI Engines import kar rahe hain (No more mock data!)
from app.services.trap_engine import generate_technical_trap, generate_behavioral_trap
from app.services.evaluation_engine import evaluate_trap_answer

router = APIRouter()

# ==========================================
# PYDANTIC SCHEMAS (Request/Response Models)
# ==========================================
class StartInterviewRequest(BaseModel):
    candidate_id: str
    interview_type: str = "technical"  # "technical" ya "behavioral" pass kar sakte ho

class SubmitAnswerRequest(BaseModel):
    candidate_id: str
    original_question: str
    candidate_answer: str
    attempt_number: int  # 1 for first trap, 2 for the double-down follow-up

# ==========================================
# ENDPOINTS
# ==========================================
@router.post("/start")
def start_interview(req: StartInterviewRequest):
    """
    Generates a 3-question dynamic interview (True Premise, False Premise Trap, Open Ended).
    """
    try:
        # Request ke type ke hisaab se sahi engine trigger hoga
        if req.interview_type.lower() == "behavioral":
            data = generate_behavioral_trap(req.candidate_id)
        else:
            data = generate_technical_trap(req.candidate_id)
            
        if "error" in data:
            raise HTTPException(status_code=400, detail=data["error"])
            
        return {"status": "success", "data": data}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start interview: {str(e)}")


@router.post("/answer")
def submit_answer(req: SubmitAnswerRequest):
    """
    Evaluates candidate's answer using the strict 'Double-Down' psychological protocol.
    Returns: VERIFIED, FLAGGED, FOLLOW_UP, or HUMAN_REVIEW.
    """
    try:
        # Seedha hamare naye evaluation engine ko answer bhej rahe hain
        evaluation_result = evaluate_trap_answer(
            original_question=req.original_question,
            candidate_answer=req.candidate_answer,
            attempt_number=req.attempt_number
        )
        
        if "error" in evaluation_result:
             raise HTTPException(status_code=400, detail=evaluation_result["error"])
             
        return {"status": "success", "evaluation": evaluation_result}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to evaluate answer: {str(e)}")