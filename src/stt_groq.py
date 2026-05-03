"""Speech-to-text via Groq Whisper (audio → transcript)."""

from __future__ import annotations

from groq import Groq

from src.settings import GROQ_API_KEY, MODEL_WHISPER


def transcribe_audio_bytes(
    data: bytes,
    *,
    filename: str = "audio.webm",
    api_key: str | None = None,
    model: str | None = None,
    language: str | None = "en",
) -> str:
    """Return plain transcript text. `filename` hint helps Whisper detect container codec."""
    key = (api_key or GROQ_API_KEY).strip()
    if not key:
        raise ValueError("GROQ_API_KEY is not set.")
    if not data:
        raise ValueError("Empty audio payload.")

    client = Groq(api_key=key)
    use_model = (model or MODEL_WHISPER).strip()

    tr = client.audio.transcriptions.create(
        file=(filename, data),
        model=use_model,
        language=language,
        response_format="json",
    )
    if isinstance(tr, str):
        return tr.strip()
    text = getattr(tr, "text", None)
    if text is not None:
        return str(text).strip()
    if isinstance(tr, dict):
        return str(tr.get("text") or "").strip()
    return str(tr).strip()
