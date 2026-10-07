from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TrapQuestion(BaseModel):
    """A single SkillProof interview question, including trap metadata."""

    model_config = ConfigDict(extra="forbid")

    question_id: int = Field(
        ...,
        ge=1,
        le=3,
        description="Sequential question identifier from 1 to 3.",
    )
    question_type: Literal["true_premise", "false_premise_trap", "open_ended"] = Field(
        ...,
        description=(
            "true_premise validates a claimed strength; "
            "false_premise_trap probes a weak skill as if it were a resume claim; "
            "open_ended is a messy-data judgment question."
        ),
    )
    prompt_text: str = Field(
        ...,
        min_length=1,
        description="The exact question spoken to the candidate.",
    )
    targeted_skill: str = Field(
        ...,
        min_length=1,
        description="The skill this question is designed to probe.",
    )
    is_trap: bool = Field(
        ...,
        description="Must be true only when question_type is false_premise_trap.",
    )


class InterviewOutput(BaseModel):
    """Structured Gemini payload for a three-question trap interview."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(..., min_length=1, description="Candidate identifier from the dataset.")
    strong_skill: str = Field(..., min_length=1, description="Highest-scoring skill used for the true-premise question.")
    weak_skill: str = Field(..., min_length=1, description="Lowest-scoring skill used for the false-premise trap.")
    questions: List[TrapQuestion] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Exactly three questions, one of each question_type.",
    )

    @field_validator("questions")
    @classmethod
    def validate_question_set(cls, questions: List[TrapQuestion]) -> List[TrapQuestion]:
        types = [q.question_type for q in questions]
        expected = {"true_premise", "false_premise_trap", "open_ended"}
        if set(types) != expected:
            raise ValueError(
                "questions must contain exactly one true_premise, "
                "one false_premise_trap, and one open_ended item"
            )
        for question in questions:
            if question.is_trap != (question.question_type == "false_premise_trap"):
                raise ValueError("is_trap must be true only for false_premise_trap questions")
        return questions
