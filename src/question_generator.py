"""Component 1 — Question set from JD + profile via NVIDIA NIM (OpenAI-compatible chat).

Output matches PLAN.md Question contract. Default model: openai/gpt-oss-20b.
"""

from __future__ import annotations

import json
import re
from json import JSONDecodeError
from typing import Any

from openai import OpenAI

from src.schemas import InterviewPlan
from src.settings import (
    MAX_QUESTIONS_PER_SESSION,
    NVIDIA_API_KEY,
    NVIDIA_INFERENCE_BASE_URL,
    NVIDIA_PLAN_MODEL,
    PLAN_MAX_JD_CHARS,
    PLAN_MAX_PROFILE_CHARS,
)


_PLAN_SYSTEM = f"""You are an expert hiring manager and technical interviewer.
OUTPUT ONLY valid JSON (no markdown fences, no commentary, no trailing commas) matching this shape:
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
- Escape every double quote inside string values as \\" ; never put raw newlines inside JSON strings.
"""


def _first_balanced_json_object(text: str) -> str:
    """Slice the first top-level `{ ... }` using brace depth, respecting strings (unlike rfind `}`)."""
    start = text.find("{")
    if start == -1:
        raise ValueError("Model did not return a JSON object.")
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError("Model JSON object appears truncated or has mismatched braces.")


def _strip_trailing_commas(s: str) -> str:
    """Remove `,` immediately before `}` or `]` outside of quoted strings."""
    out: list[str] = []
    in_string = False
    escape = False
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == ",":
            j = i + 1
            while j < n and s[j] in " \t\n\r":
                j += 1
            if j < n and s[j] in "}]":
                i += 1
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _normalize_unicode_quotes(s: str) -> str:
    return (
        s.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if "```" in text:
        parts = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.I)
        if parts:
            text = parts[0].strip()
    blob = _first_balanced_json_object(text)
    attempts: list[str] = [blob, _strip_trailing_commas(blob), _normalize_unicode_quotes(blob)]
    attempts.append(_strip_trailing_commas(attempts[-1]))
    last_err: Exception | None = None
    for candidate in attempts:
        try:
            return json.loads(candidate)
        except JSONDecodeError as e:
            last_err = e
            continue
    preview = text[:900] + ("…" if len(text) > 900 else "")
    hint = f" JSON parse failed ({last_err!s}). Model output preview: {preview}"
    raise ValueError(hint) from last_err


def _truncate_for_plan(jd: str, profile: str) -> tuple[str, str]:
    jd = jd.strip()
    profile = profile.strip()
    if len(jd) > PLAN_MAX_JD_CHARS:
        jd = jd[: PLAN_MAX_JD_CHARS].rstrip() + "\n[…truncated for API size limits; shorten JD or raise PLAN_MAX_JD_CHARS…]"
    if len(profile) > PLAN_MAX_PROFILE_CHARS:
        profile = (
            profile[: PLAN_MAX_PROFILE_CHARS].rstrip()
            + "\n[…truncated for API size limits; shorten profile or raise PLAN_MAX_PROFILE_CHARS…]"
        )
    return jd, profile


def _nvidia_chat_plan_json(
    *,
    api_key: str,
    base_url: str,
    model: str,
    system: str,
    user: str,
) -> str:
    client = OpenAI(base_url=base_url.rstrip("/"), api_key=api_key)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    common: dict[str, Any] = {
        "model": model,
        "temperature": 0.35,
        # Large max_tokens inflates TPM / request estimates; interview plan JSON is usually <4k tokens.
        "max_tokens": 4096,
        "messages": messages,
    }
    try:
        completion = client.chat.completions.create(
            **common,
            response_format={"type": "json_object"},
        )
    except Exception:
        completion = client.chat.completions.create(**common)
    choice = completion.choices[0].message
    raw = (getattr(choice, "content", None) or "").strip()
    if not raw:
        reasoning = getattr(choice, "reasoning_content", None) or getattr(choice, "reasoning", None)
        if reasoning:
            raw = str(reasoning).strip()
    if not raw:
        raise ValueError("NVIDIA model returned empty content for plan generation.")
    return raw


def generate_interview_plan(
    job_description: str,
    candidate_profile: str,
    *,
    nvidia_api_key: str | None = None,
    model: str | None = None,
) -> InterviewPlan:
    key = (nvidia_api_key or NVIDIA_API_KEY).strip()
    if not key:
        raise ValueError(
            "NVIDIA_API_KEY is not set. Add it to .env or paste it under “NVIDIA (plan)” on the credentials screen.",
        )
    use_model = (model or NVIDIA_PLAN_MODEL).strip()
    base = (NVIDIA_INFERENCE_BASE_URL or "https://integrate.api.nvidia.com/v1").strip()

    jd, prof = _truncate_for_plan(job_description, candidate_profile)
    user = f"""JOB DESCRIPTION:\n{jd}\n\nCANDIDATE PROFILE:\n{prof}\n"""
    raw = _nvidia_chat_plan_json(
        api_key=key,
        base_url=base,
        model=use_model,
        system=_PLAN_SYSTEM,
        user=user,
    )
    data = _extract_json_object(raw)
    plan = InterviewPlan.model_validate(data)
    plan.questions.sort(key=lambda q: q.order)
    return plan
