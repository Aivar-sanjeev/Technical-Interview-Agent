"""Component 2 — Live interviewer text (NVIDIA NIM, OpenAI-compatible streaming)."""

from __future__ import annotations

import json
from collections.abc import Iterator

from openai import OpenAI

from src.policies import policy_prefix_for_event, should_nudge_wrap
from src.schemas import InterviewPlan, Transcript, TranscriptTurn
from src.settings import MAX_FOLLOW_UPS, NVIDIA_API_KEY, NVIDIA_INFERENCE_BASE_URL, NVIDIA_INTERVIEW_MODEL


_CONDUCTOR_SYSTEM = """You are a senior technical interviewer conducting a live voice interview.

Goals:
- Sound human, concise, and conversational. Prefer short spoken sentences.
- Ask the active question, listen, then choose follow-ups using depth_probes when appropriate.
- If the candidate asks to clarify: rephrase or narrow scope only — no teaching or partial solutions.
- Never hint at a correct answer, algorithm steps, or what they "should" say.
- Never coach toward content (no "you might want to mention X" where X reveals the answer).

Classification awareness (do not read labels aloud):
- If they are off-topic: redirect once politely, then offer to move on.
- If they are silent or stuck: encourage thinking aloud or clarifying the question wording only.

If interaction is \"opening\", greet briefly then ask the active question clearly.

If you receive a system note about [TIME_LIMIT_REACHED], acknowledge briefly and transition to the next question angle without solving anything."""


def _serialize_transcript(tr: Transcript, max_turns: int = 24) -> str:
    tail = tr.turns[-max_turns:] if len(tr.turns) > max_turns else tr.turns
    lines = []
    for t in tail:
        lines.append(
            {
                "id": t.id,
                "role": t.role,
                "text": t.text,
                "classification": t.classification,
                "depth_signal": t.depth_signal,
            }
        )
    return json.dumps(lines, ensure_ascii=False)


def _active_question(plan: InterviewPlan, q_index: int) -> dict:
    if q_index < 0 or q_index >= len(plan.questions):
        return {"error": "no_more_questions"}
    q = plan.questions[q_index]
    return {
        "id": q.id,
        "topic": q.topic,
        "question": q.question,
        "depth_probes": [p.model_dump() for p in q.depth_probes],
        "eval_criteria": q.eval_criteria,
        "difficulty": q.difficulty,
        "order": q.order,
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
    prefix += should_nudge_wrap(followups_on_current, MAX_FOLLOW_UPS)
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
    key = (api_key or NVIDIA_API_KEY).strip()
    if not key:
        raise ValueError("NVIDIA_API_KEY is not set.")

    if q_index < 0 or q_index >= len(plan.questions):
        yield "We have reached the end of the planned question set. Thank the candidate warmly and end the interview."
        return

    client = OpenAI(base_url=NVIDIA_INFERENCE_BASE_URL.rstrip("/"), api_key=key)
    use_model = (model or NVIDIA_INTERVIEW_MODEL).strip()

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
        max_tokens=2048,
        messages=[
            {"role": "system", "content": _CONDUCTOR_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        stream=True,
    )

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content or ""
        if delta:
            yield delta


def opening_message(plan: InterviewPlan, q_index: int) -> str:
    q = plan.questions[q_index]
    return (
        f"Hi — thanks for joining. I'm interviewing for {plan.role_title or 'this role'}. "
        f"Let's start with: {q.question}"
    )


def transcript_with_turn(tr: Transcript, turn: TranscriptTurn) -> Transcript:
    new_turns = list(tr.turns)
    new_turns.append(turn)
    return Transcript(turns=new_turns)
