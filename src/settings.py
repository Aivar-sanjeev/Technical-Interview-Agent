"""Single source of truth for NVIDIA NIM / Riva settings and interview timing."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

# NVIDIA API key (integrate.api.nvidia.com + Riva ASR metadata)
NVIDIA_API_KEY: str = (os.getenv("NVIDIA_API_KEY") or "").strip()

# Speech-to-text: Nemotron ASR streaming (Riva gRPC) only.
NVIDIA_ASR_FUNCTION_ID: str = (
    os.getenv("NVIDIA_ASR_FUNCTION_ID") or "bb0837de-8c7b-481f-9ec8-ef5663e9c1fa"
).strip()
NVIDIA_ASR_GRPC_URI: str = (os.getenv("NVIDIA_ASR_GRPC_URI") or "grpc.nvcf.nvidia.com:443").strip()
NVIDIA_ASR_MODEL: str = (os.getenv("NVIDIA_ASR_MODEL") or "").strip()

# OpenAI-compatible chat at NVIDIA NIM
NVIDIA_INFERENCE_BASE_URL: str = (
    os.getenv("NVIDIA_INFERENCE_BASE_URL") or "https://integrate.api.nvidia.com/v1"
).strip()
NVIDIA_PLAN_MODEL: str = (os.getenv("NVIDIA_PLAN_MODEL") or "openai/gpt-oss-20b").strip()
NVIDIA_INTERVIEW_MODEL: str = (os.getenv("NVIDIA_INTERVIEW_MODEL") or "openai/gpt-oss-20b").strip()
NVIDIA_EVAL_MODEL: str = (os.getenv("NVIDIA_EVAL_MODEL") or "openai/gpt-oss-20b").strip()

PLAN_MAX_JD_CHARS: int = int(os.getenv("PLAN_MAX_JD_CHARS", "6000"))
PLAN_MAX_PROFILE_CHARS: int = int(os.getenv("PLAN_MAX_PROFILE_CHARS", "4500"))

QUESTION_TIMEOUT_SECONDS: int = int(os.getenv("QUESTION_TIMEOUT_SECONDS", "120"))
SILENCE_THRESHOLD_SECONDS: float = float(os.getenv("SILENCE_THRESHOLD_SECONDS", "4"))
MAX_FOLLOW_UPS: int = int(os.getenv("MAX_FOLLOW_UPS", "2"))
MAX_RETRIES_OFF_TOPIC: int = int(os.getenv("MAX_RETRIES_OFF_TOPIC", "1"))
MAX_QUESTIONS_PER_SESSION: int = int(os.getenv("MAX_QUESTIONS_PER_SESSION", "12"))

APP_NAME: str = os.getenv("APP_NAME", "TechInterviewAgent").strip()

VOICE_SILENCE_NUDGE_SECONDS: float = float(os.getenv("VOICE_SILENCE_NUDGE_SECONDS", "22"))
VOICE_SILENCE_SKIP_SECONDS: float = float(os.getenv("VOICE_SILENCE_SKIP_SECONDS", "28"))
