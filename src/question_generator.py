"""Generate InterviewPlan from job description + candidate profile (Groq only).

This module does not read live transcripts and does not evaluate answers.
"""

from __future__ import annotations

import json
import re
from typing import Any

from groq import Groq

from src.config import GROQ_API_KEY, MODEL_PLAN
from src.schemas import InterviewPlan


_PLAN_SYSTEM = """You are an expert hiring manager and technical interviewer.
Your job is to OUTPUT ONLY valid JSON (no markdown fences, no commentary) matching this shape:
{
  "version": "1",
  "role_title": string,
  "role_summary": string (2-4 sentences),
  "assumed_seniority": "junior" | "mid" | "senior" | "staff",
  "key_skills_from_jd": string[] (5-12 items),
  "sections": string[] (ordered section headers),
  "questions": [
    {
      "id": "q-1",
      "stem": string (clear interview question),
      "intent": string (what you are testing),
      "difficulty": "easy" | "medium" | "hard",
      "section": string (must match one of sections),
      "follow_up_hooks": string[] (2-4 short probe ideas, NOT answers),
      "must_cover": boolean
    }
  ]
}

Rules:
- Produce 10-14 questions total, spread across sections (systems, coding/problem-solving, role-specific depth, collaboration/ownership as appropriate).
- Tailor difficulty and topics to the CANDIDATE PROFILE (their background) and the JOB DESCRIPTION.
- Questions must be fair, professional, and free of trick trivia unless the JD emphasizes it.
- follow_up_hooks must be probe angles only — never hints or solutions.
- Use ids q-1, q-2, ... in order.
"""


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if "```" in text:
        parts = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.I)
        if parts:
            text = parts[0].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model did not return a JSON object.")
    return json.loads(text[start : end + 1])


def generate_interview_plan(
    job_description: str,
    candidate_profile: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
) -> InterviewPlan:
    key = (api_key or GROQ_API_KEY).strip()
    if not key:
        raise ValueError("GROQ_API_KEY is not set.")

    client = Groq(api_key=key)
    use_model = (model or MODEL_PLAN).strip()

    user = f"""JOB DESCRIPTION:\n{job_description.strip()}\n\nCANDIDATE PROFILE:\n{candidate_profile.strip()}\n"""

    completion = client.chat.completions.create(
        model=use_model,
        temperature=0.35,
        messages=[
            {"role": "system", "content": _PLAN_SYSTEM},
            {"role": "user", "content": user},
        ],
    )
    raw = (completion.choices[0].message.content or "").strip()
    data = _extract_json_object(raw)
    return InterviewPlan.model_validate(data)
