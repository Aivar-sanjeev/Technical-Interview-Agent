"""Shared data contracts for plan generation, interview sessions, and evaluation.

Other modules depend only on these types — not on each other's implementation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PlanQuestion(BaseModel):
    id: str = Field(description="Stable id, e.g. q-1")
    stem: str = Field(description="Main question text as asked in interview")
    intent: str = Field(description="What signal this question probes")
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    section: str = Field(description="Section name, e.g. System design")
    follow_up_hooks: list[str] = Field(
        default_factory=list,
        description="Angles for follow-ups — not full scripted answers",
    )
    must_cover: bool = False


class InterviewPlan(BaseModel):
    version: str = "1"
    role_title: str = ""
    role_summary: str = ""
    assumed_seniority: str = "mid"
    key_skills_from_jd: list[str] = Field(default_factory=list)
    sections: list[str] = Field(default_factory=list)
    questions: list[PlanQuestion] = Field(default_factory=list)


class TranscriptTurn(BaseModel):
    id: str
    role: Literal["interviewer", "candidate", "system"]
    text: str
    ts: float = 0.0


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
