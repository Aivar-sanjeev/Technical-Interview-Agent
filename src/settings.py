"""Single source of truth for model IDs and interview timing (Groq-only stack).

Aligned with PLAN.md — swap models here only.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

# API
GROQ_API_KEY: str = (os.getenv("GROQ_API_KEY") or "").strip()

# ── Groq model IDs ─────────────────────────────────────────────
MODEL_PLAN: str = os.getenv("GROQ_MODEL_PLAN", "llama-3.3-70b-versatile").strip()
MODEL_INTERVIEW: str = os.getenv("GROQ_MODEL_INTERVIEW", "llama-3.3-70b-versatile").strip()
MODEL_EVAL: str = os.getenv("GROQ_MODEL_EVAL", "llama-3.3-70b-versatile").strip()
MODEL_WHISPER: str = os.getenv("GROQ_MODEL_WHISPER", "whisper-large-v3").strip()
MODEL_TTS: str = os.getenv("GROQ_MODEL_TTS", "canopylabs/orpheus-v1-english").strip()
ORPHEUS_VOICE: str = os.getenv("GROQ_ORPHEUS_VOICE", "troy").strip()
ORPHEUS_MAX_INPUT_CHARS: int = int(os.getenv("GROQ_ORPHEUS_MAX_CHARS", "200"))

# ── Interview timing (PLAN.md) ─────────────────────────────────
QUESTION_TIMEOUT_SECONDS: int = int(os.getenv("QUESTION_TIMEOUT_SECONDS", "40"))
SILENCE_THRESHOLD_SECONDS: float = float(os.getenv("SILENCE_THRESHOLD_SECONDS", "4"))
MAX_FOLLOW_UPS: int = int(os.getenv("MAX_FOLLOW_UPS", "2"))
MAX_RETRIES_OFF_TOPIC: int = int(os.getenv("MAX_RETRIES_OFF_TOPIC", "1"))
MAX_QUESTIONS_PER_SESSION: int = int(os.getenv("MAX_QUESTIONS_PER_SESSION", "12"))

APP_NAME: str = os.getenv("APP_NAME", "TechInterviewAgent").strip()
