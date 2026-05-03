"""Post-session structured evaluation (NVIDIA NIM, OpenAI-compatible chat)."""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from src.question_generator import _extract_json_object
from src.settings import NVIDIA_API_KEY, NVIDIA_EVAL_MODEL, NVIDIA_INFERENCE_BASE_URL
from src.schemas import EvaluationReport, InterviewPlan, Transcript


_EVAL_SYSTEM = """You are an experienced hiring committee reviewer.

OUTPUT ONLY valid JSON (no markdown fences, no commentary) with this exact shape:
{
  "version": "1",
  "recommendation": "strong_hire" | "hire" | "maybe" | "no_hire",
  "overall_summary": string (3-6 sentences, factual),
  "strengths": string[] (3-6 bullets),
  "gaps": string[] (3-6 bullets),
  "red_flags": string[] (0-4 items; empty array if none),
  "scenario_handling_notes": string (briefly note silence/clarify/off-topic handling if visible in transcript),
  "competencies": [
    {
      "competency": string,
      "score": integer 1-5,
      "confidence": "low" | "medium" | "high",
      "summary": string,
      "evidence": [ { "quote": string, "turn_id": string } ]
    }
  ]
}

Rules:
- Produce a **cumulative, holistic** assessment across the **entire** interview: every question and follow-up in the plan may inform judgment, but the recommendation must reflect performance **over the full session**, not isolated answers only.
- Ground every competency row in evidence quotes taken VERBATIM from CANDIDATE turns in the transcript JSON. Prefer quotes from different parts of the interview when possible. Quotes must be exact substrings of candidate text (short phrases to 1-2 sentences).
- turn_id must match the transcript turn id for that quote when possible; if unknown use "".
- If the transcript lacks evidence for a competency, score conservatively and include at most one evidence item explaining insufficiency.
- Do not invent candidate statements; quotes must appear in the transcript.
- Do not coach or rewrite the interview; evaluate only.
"""


def _candidate_corpus(transcript: Transcript) -> dict[str, str]:
    """Map turn id -> text for candidate turns; plus concatenated blob for substring checks."""
    by_id: dict[str, str] = {}
    parts: list[str] = []
    for t in transcript.turns:
        if t.role == "candidate":
            by_id[t.id] = t.text
            parts.append(t.text)
    return {"by_id": by_id, "blob": "\n".join(parts)}


def validate_report_against_transcript(report: EvaluationReport, transcript: Transcript) -> list[str]:
    """Return human-readable issues when quotes are missing or not grounded."""
    issues: list[str] = []
    corp = _candidate_corpus(transcript)
    blob = corp["blob"]
    by_id = corp["by_id"]

    for comp in report.competencies:
        if not comp.evidence:
            issues.append(f"[{comp.competency}] has no evidence items.")
            continue
        for ev in comp.evidence:
            q = (ev.quote or "").strip()
            if len(q) < 8:
                issues.append(f"[{comp.competency}] quote too short or empty: {q!r}")
                continue
            if q not in blob:
                issues.append(f"[{comp.competency}] quote not found verbatim in candidate turns: {q[:120]!r}...")
            if ev.turn_id and ev.turn_id in by_id and q not in by_id[ev.turn_id]:
                issues.append(
                    f"[{comp.competency}] quote not contained in cited turn_id {ev.turn_id!r}."
                )
    return issues


def evaluate_interview(
    *,
    job_description: str,
    candidate_profile: str,
    plan: InterviewPlan,
    transcript: Transcript,
    api_key: str | None = None,
    model: str | None = None,
) -> EvaluationReport:
    key = (api_key or NVIDIA_API_KEY).strip()
    if not key:
        raise ValueError("NVIDIA_API_KEY is not set.")

    client = OpenAI(base_url=NVIDIA_INFERENCE_BASE_URL.rstrip("/"), api_key=key)
    use_model = (model or NVIDIA_EVAL_MODEL).strip()

    payload = {
        "job_description": job_description.strip(),
        "candidate_profile": candidate_profile.strip(),
        "interview_plan": plan.model_dump(),
        "transcript": [t.model_dump() for t in transcript.turns],
    }
    user = json.dumps(payload, ensure_ascii=False)

    completion = client.chat.completions.create(
        model=use_model,
        temperature=0.2,
        max_tokens=8192,
        messages=[
            {"role": "system", "content": _EVAL_SYSTEM},
            {"role": "user", "content": user},
        ],
    )
    choice = completion.choices[0].message
    raw = (getattr(choice, "content", None) or "").strip()
    if not raw:
        reasoning = getattr(choice, "reasoning_content", None) or getattr(choice, "reasoning", None)
        if reasoning:
            raw = str(reasoning).strip()
    if not raw:
        raise ValueError("NVIDIA model returned empty content for evaluation.")
    data = _extract_json_object(raw)
    return EvaluationReport.model_validate(data)


def evaluate_with_repair(
    *,
    job_description: str,
    candidate_profile: str,
    plan: InterviewPlan,
    transcript: Transcript,
    api_key: str | None = None,
    model: str | None = None,
    max_repairs: int = 1,
) -> tuple[EvaluationReport, list[str]]:
    """Run evaluation; if evidence validation fails, one repair pass with issues fed back."""
    report = evaluate_interview(
        job_description=job_description,
        candidate_profile=candidate_profile,
        plan=plan,
        transcript=transcript,
        api_key=api_key,
        model=model,
    )
    issues = validate_report_against_transcript(report, transcript)
    if not issues or max_repairs <= 0:
        return report, issues

    key = (api_key or NVIDIA_API_KEY).strip()
    client = OpenAI(base_url=NVIDIA_INFERENCE_BASE_URL.rstrip("/"), api_key=key)
    use_model = (model or NVIDIA_EVAL_MODEL).strip()
    repair_user = json.dumps(
        {
            "previous_json": report.model_dump(),
            "validation_issues": issues,
            "transcript": [t.model_dump() for t in transcript.turns],
            "instruction": "Return corrected JSON only. Fix evidence quotes to be exact substrings from candidate turns only; fix turn_id to match.",
        },
        ensure_ascii=False,
    )
    completion = client.chat.completions.create(
        model=use_model,
        temperature=0.1,
        max_tokens=8192,
        messages=[
            {"role": "system", "content": _EVAL_SYSTEM},
            {"role": "user", "content": repair_user},
        ],
    )
    choice = completion.choices[0].message
    raw = (getattr(choice, "content", None) or "").strip()
    if not raw:
        reasoning = getattr(choice, "reasoning_content", None) or getattr(choice, "reasoning", None)
        if reasoning:
            raw = str(reasoning).strip()
    if not raw:
        return report, issues
    data = _extract_json_object(raw)
    report2 = EvaluationReport.model_validate(data)
    issues2 = validate_report_against_transcript(report2, transcript)
    return report2, issues2
