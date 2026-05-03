"""Contracts for plan generation, live interview, and evaluation (PLAN-aligned questions).

Components share these types only — no cross-imports of implementation.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class DepthProbe(BaseModel):
    probe: str
    listen_for: str = Field(default="", description="Internal signal — not shown to candidate verbatim")


class PlanQuestion(BaseModel):
    """PLAN.md Question shape — Component 1 output / Component 2 input."""

    id: str
    topic: str
    question: str = Field(description="Exact interview question wording")
    depth_probes: Annotated[list[DepthProbe], Field(max_length=2)] = Field(default_factory=list)
    eval_criteria: list[str] = Field(default_factory=list)
    difficulty: Literal["junior", "mid", "senior", "staff"] = "mid"
    order: int = 0


class InterviewPlan(BaseModel):
    version: str = "1"
    role_title: str = ""
    role_summary: str = ""
    assumed_seniority: str = "mid"
    key_skills_from_jd: list[str] = Field(default_factory=list)
    sections: list[str] = Field(default_factory=list, description="Optional UX grouping from generator")
    questions: list[PlanQuestion] = Field(default_factory=list)


class TranscriptTurn(BaseModel):
    id: str
    role: Literal["interviewer", "candidate", "system"]
    text: str
    ts: float = 0.0
    question_id: str | None = None
    classification: Literal["ANSWER", "SKIP", "OFF_TOPIC", "CLARIFICATION", "UNKNOWN"] | None = None
    depth_signal: Literal["STRONG", "ADEQUATE", "WEAK", "NONE"] | None = None


class Transcript(BaseModel):
    turns: list[TranscriptTurn] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    quote: str = Field(description="Verbatim substring from candidate turns")
    turn_id: str = ""


class CompetencyScore(BaseModel):
    competency: str
    score: int = Field(ge=1, le=5)
    confidence: Literal["low", "medium", "high"] = "medium"
    summary: str = ""
    evidence: list[EvidenceItem] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    version: str = "1"
    recommendation: Literal["strong_hire", "hire", "maybe", "no_hire"] = "maybe"
    overall_summary: str = ""
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    competencies: list[CompetencyScore] = Field(default_factory=list)
    scenario_handling_notes: str = ""
