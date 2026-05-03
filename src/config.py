"""Environment-driven settings for Groq models and API key."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

GROQ_API_KEY: str = (os.getenv("GROQ_API_KEY") or "").strip()
MODEL_PLAN: str = os.getenv("GROQ_MODEL_PLAN", "llama-3.3-70b-versatile").strip()
MODEL_INTERVIEW: str = os.getenv("GROQ_MODEL_INTERVIEW", "llama-3.3-70b-versatile").strip()
MODEL_EVAL: str = os.getenv("GROQ_MODEL_EVAL", "llama-3.3-70b-versatile").strip()
