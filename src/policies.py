"""Deterministic interview policy hints (no LLM).

The conductor LLM receives these strings as context; behavior stays testable without Groq.
"""

from __future__ import annotations

from src.settings import MAX_FOLLOW_UPS


def policy_prefix_for_event(event: str) -> str:
    if event == "silence":
        return (
            "[Situation: The candidate has gone silent or not responded in time. "
            "Acknowledge briefly, encourage thinking aloud or asking for clarification of the question wording only, "
            "and offer to move on if they are stuck. Do NOT outline a solution, algorithm steps, or 'correct' answer. "
            "Keep to 2–4 short sentences.]\n\n"
        )
    if event == "clarify_request":
        return (
            "[Situation: The candidate asked to clarify the question. "
            "Rephrase the question, narrow scope, or disambiguate definitions. "
            "Do NOT teach, hint at the solution, or give examples that reveal the answer.]\n\n"
        )
    if event == "off_topic":
        return (
            "[Situation: The candidate answered off-topic or drifted from the question. "
            "Politely redirect to the question's intent. Do NOT answer the question for them. "
            "Offer one chance to refocus; if still off, note you will move on after their next reply.]\n\n"
        )
    return ""


def should_nudge_wrap(followups_on_current: int, max_followups: int | None = None) -> str:
    cap = max_followups if max_followups is not None else MAX_FOLLOW_UPS
    if followups_on_current < cap:
        return ""
    return (
        "[Guidance: You have already probed this question several times. "
        "Briefly acknowledge their effort, then transition to the next topic or closing wrap-up "
        "unless they are mid-thought in this message.]\n\n"
    )
