"""Component 1 — Question set from JD + profile (Groq JSON only).

No interview or evaluation logic. Output matches PLAN.md Question contract.
"""

from __future__ import annotations

import json
import re
from typing import Any

from groq import Groq

from src.schemas import InterviewPlan
from src.settings import GROQ_API_KEY, MAX_QUESTIONS_PER_SESSION, MODEL_PLAN


_PLAN_SYSTEM = f"""You are an expert hiring manager and technical interviewer.
OUTPUT ONLY valid JSON (no markdown fences, no commentary) matching this shape:
{{
  "version": "1",
  "role_title": string,
  "role_summary": string (2-4 sentences),
  "assumed_seniority": "junior" | "mid" | "senior" | "staff",
  "key_skills_from_jd": string[] (5-12 items),
  "sections": string[] (optional section headers for UX),
  "questions": [
    {{
      "id": "q-1",
      "topic": string (short topic label, e.g. \"API design\"),
      "question": string (exact interview question wording),
      "depth_probes": [{{"probe": string, "listen_for": string}}],
      "eval_criteria": string[] (signals for later evaluation),
      "difficulty": "junior" | "mid" | "senior" | "staff",
      "order": 1
    }}
  ]
}}

Rules:
- Produce 6-{MAX_QUESTIONS_PER_SESSION} questions, strictly increasing order starting at 1.
- At most 2 depth_probes per question; probes are follow-up questions only — NEVER hints or answers.
- listen_for is an internal rubric phrase (what a strong answer might mention) — not shown to candidate as a checklist.
- Tailor topics and difficulty to the candidate profile and JD.
- eval_criteria must be short observable signals (no solution leakage).
- Questions must be fair and professional.
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
    plan = InterviewPlan.model_validate(data)
    plan.questions.sort(key=lambda q: q.order)
    return plan
