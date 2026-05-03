"""Decode browser / upload audio to 16 kHz mono s16le PCM for Riva ASR."""

from __future__ import annotations

import io
import shutil
import subprocess
import wave

TARGET_RATE_HZ = 16000


def bytes_to_pcm_s16le_mono_16k(data: bytes) -> bytes:
    """Return raw PCM (16-bit little-endian, mono, 16 kHz). Uses ffmpeg unless input is already that WAV."""
    if not data:
        raise ValueError("Empty audio payload.")
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        try:
            with wave.open(io.BytesIO(data), "rb") as wf:
                if (
                    wf.getsampwidth() == 2
                    and wf.getnchannels() == 1
                    and wf.getframerate() == TARGET_RATE_HZ
                ):
                    return wf.readframes(wf.getnframes())
        except Exception:
            pass
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValueError(
            "NVIDIA STT needs ffmpeg in PATH to decode WebM/Opus from the browser. "
            "Install ffmpeg, or record WAV mono 16-bit 16 kHz for Nemotron ASR."
        )
    proc = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            str(TARGET_RATE_HZ),
            "pipe:1",
        ],
        input=data,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", errors="replace")[:800]
        raise ValueError(f"ffmpeg could not decode audio (exit {proc.returncode}): {err}")
    if not proc.stdout:
        raise ValueError("ffmpeg produced empty PCM output.")
    return proc.stdout
