"""Real-time interview conductor (Groq streaming only).

Does not generate the initial question set and does not produce final evaluations.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from groq import Groq

from src.config import GROQ_API_KEY, MODEL_INTERVIEW
from src.policies import policy_prefix_for_event, should_nudge_wrap
from src.schemas import InterviewPlan, Transcript, TranscriptTurn


_CONDUCTOR_SYSTEM = """You are a senior technical interviewer conducting a live interview.

Goals:
- Sound human, concise, and conversational (not a form or checklist read aloud).
- Ask the active question, listen, then choose appropriate follow-ups: go deeper on strong answers; help clarify scope on vague answers; do not lecture.
- You may rephrase or scope the question if the candidate asks for clarification.
- Never reveal or hint at a "correct" answer, algorithm, or solution sketch. Do not name specific design choices they "should" pick.
- Never coach the candidate toward an answer (no "you might want to mention X" where X is content of the solution).

Style:
- Short paragraphs; prefer one question at a time.
- If the candidate is stuck, you may offer to move to another angle or the next topic — without solving the problem.

You will receive JSON describing the interview plan excerpt, transcript, active question, and optional situation notes. Respond as the interviewer only.

If the field interaction is \"opening\", you are either starting the interview or transitioning to a new main question: greet briefly (if appropriate), then ask the active question clearly — do not invent that the candidate already answered it."""


def _serialize_transcript(tr: Transcript, max_turns: int = 24) -> str:
    tail = tr.turns[-max_turns:] if len(tr.turns) > max_turns else tr.turns
    lines = []
    for t in tail:
        lines.append({"id": t.id, "role": t.role, "text": t.text})
    return json.dumps(lines, ensure_ascii=False)


def _active_question(plan: InterviewPlan, q_index: int) -> dict:
    if q_index < 0 or q_index >= len(plan.questions):
        return {"error": "no_more_questions", "stem": "", "intent": "", "hooks": []}
    q = plan.questions[q_index]
    return {
        "id": q.id,
        "stem": q.stem,
        "intent": q.intent,
        "difficulty": q.difficulty,
        "section": q.section,
        "follow_up_hooks": q.follow_up_hooks,
    }


def build_user_payload(
    plan: InterviewPlan,
    transcript: Transcript,
    q_index: int,
    *,
    candidate_message: str,
    event_type: str,
    followups_on_current: int,
    interaction: str = "continue",
) -> str:
    prefix = policy_prefix_for_event(event_type)
    prefix += should_nudge_wrap(followups_on_current)
    payload = {
        "role_title": plan.role_title,
        "role_summary": plan.role_summary,
        "assumed_seniority": plan.assumed_seniority,
        "active_question": _active_question(plan, q_index),
        "question_progress": f"{q_index + 1} of {len(plan.questions)}",
        "recent_transcript": json.loads(_serialize_transcript(transcript)),
        "latest_candidate_message": candidate_message,
        "interaction": interaction,
    }
    return prefix + json.dumps(payload, ensure_ascii=False)


def stream_interviewer_reply(
    plan: InterviewPlan,
    transcript: Transcript,
    q_index: int,
    *,
    candidate_message: str,
    event_type: str = "normal",
    followups_on_current: int = 0,
    interaction: str = "continue",
    api_key: str | None = None,
    model: str | None = None,
) -> Iterator[str]:
    key = (api_key or GROQ_API_KEY).strip()
    if not key:
        raise ValueError("GROQ_API_KEY is not set.")

    if q_index < 0 or q_index >= len(plan.questions):
        yield "We have reached the end of the planned question set. Thank the candidate warmly and end the interview."
        return

    client = Groq(api_key=key)
    use_model = (model or MODEL_INTERVIEW).strip()

    user_content = build_user_payload(
        plan,
        transcript,
        q_index,
        candidate_message=candidate_message,
        event_type=event_type,
        followups_on_current=followups_on_current,
        interaction=interaction,
    )

    stream = client.chat.completions.create(
        model=use_model,
        temperature=0.55,
        messages=[
            {"role": "system", "content": _CONDUCTOR_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            yield delta


def opening_message(plan: InterviewPlan, q_index: int) -> str:
    """Non-streamed fallback (unused if streaming opening)."""
    q = plan.questions[q_index]
    return (
        f"Hi — thanks for joining. I'm interviewing for {plan.role_title or 'this role'}. "
        f"Let's start with: {q.stem}"
    )


def transcript_with_turn(tr: Transcript, turn: TranscriptTurn) -> Transcript:
    new_turns = list(tr.turns)
    new_turns.append(turn)
    return Transcript(turns=new_turns)
