"""Speech-to-text: NVIDIA Nemotron ASR streaming (Riva gRPC) only."""

from __future__ import annotations

from src.settings import NVIDIA_API_KEY


def transcribe_audio_bytes(
    data: bytes,
    *,
    filename: str = "audio.webm",
    api_key: str | None = None,
    nvidia_api_key: str | None = None,
    language: str | None = "en",
) -> str:
    _ = language  # Nemotron English; language hint reserved
    nv = (nvidia_api_key or NVIDIA_API_KEY or "").strip()
    if not nv:
        raise ValueError(
            "NVIDIA API key missing for Nemotron ASR. Set NVIDIA_API_KEY in .env or paste it in the UI.",
        )
    from src.stt_nvidia import transcribe_nvidia_bytes

    return transcribe_nvidia_bytes(data, filename_hint=filename, api_key=nv)
