"""Text-to-speech via Groq Orpheus (chunked — API max input length applies per call)."""

from __future__ import annotations

import re
from typing import Iterator

from groq import Groq

from src.settings import GROQ_API_KEY, MODEL_TTS, ORPHEUS_MAX_INPUT_CHARS, ORPHEUS_VOICE


def _split_for_orpheus(text: str, max_chars: int) -> list[str]:
    """Split into chunks <= max_chars, preferring sentence boundaries."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    max_chars = max(32, min(max_chars, ORPHEUS_MAX_INPUT_CHARS))
    parts: list[str] = []
    buf = ""
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if not sent:
            continue
        if len(buf) + len(sent) + 1 <= max_chars:
            buf = f"{buf} {sent}".strip() if buf else sent
        else:
            if buf:
                parts.append(buf)
            if len(sent) <= max_chars:
                buf = sent
            else:
                for i in range(0, len(sent), max_chars):
                    chunk = sent[i : i + max_chars].strip()
                    if chunk:
                        parts.append(chunk)
                buf = ""
    if buf:
        parts.append(buf)
    return parts


def synthesize_speech_wav_chunks(
    text: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
    voice: str | None = None,
    max_chars: int | None = None,
) -> Iterator[bytes]:
    """Yield WAV bytes per Orpheus-safe chunk (sequential playback)."""
    key = (api_key or GROQ_API_KEY).strip()
    if not key:
        raise ValueError("GROQ_API_KEY is not set.")

    client = Groq(api_key=key)
    use_model = (model or MODEL_TTS).strip()
    use_voice = (voice or ORPHEUS_VOICE).strip()
    cap = max_chars or ORPHEUS_MAX_INPUT_CHARS

    for chunk in _split_for_orpheus(text, cap):
        speech = client.audio.speech.create(
            model=use_model,
            voice=use_voice,
            input=chunk[:ORPHEUS_MAX_INPUT_CHARS],
            response_format="wav",
        )
        read_fn = getattr(speech, "read", None)
        if callable(read_fn):
            data = read_fn()
            if isinstance(data, (bytes, bytearray)):
                yield bytes(data)
            else:
                yield bytes(speech)
        elif hasattr(speech, "content"):
            yield bytes(speech.content)  # type: ignore[attr-defined]
        else:
            yield bytes(speech)
