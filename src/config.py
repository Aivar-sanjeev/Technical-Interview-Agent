"""Backward-compatible re-exports — prefer `src.settings` for new code."""

from src.settings import (  # noqa: F401
    APP_NAME,
    MAX_FOLLOW_UPS,
    MAX_QUESTIONS_PER_SESSION,
    MAX_RETRIES_OFF_TOPIC,
    NVIDIA_API_KEY,
    NVIDIA_EVAL_MODEL,
    NVIDIA_INTERVIEW_MODEL,
    NVIDIA_PLAN_MODEL,
    QUESTION_TIMEOUT_SECONDS,
    SILENCE_THRESHOLD_SECONDS,
)
