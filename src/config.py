"""Backward-compatible re-exports — prefer `src.settings` for new code."""

from src.settings import (  # noqa: F401
    APP_NAME,
    GROQ_API_KEY,
    MAX_FOLLOW_UPS,
    MAX_QUESTIONS_PER_SESSION,
    MAX_RETRIES_OFF_TOPIC,
    MODEL_EVAL,
    MODEL_INTERVIEW,
    MODEL_PLAN,
    MODEL_TTS,
    MODEL_WHISPER,
    ORPHEUS_MAX_INPUT_CHARS,
    ORPHEUS_VOICE,
    QUESTION_TIMEOUT_SECONDS,
    SILENCE_THRESHOLD_SECONDS,
)
