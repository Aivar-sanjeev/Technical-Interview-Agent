"""Speech-to-text via NVIDIA Nemotron ASR (Riva gRPC on grpc.nvcf.nvidia.com).

Nemotron ASR streaming is streaming-only; we send chunked PCM and aggregate final hypotheses.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from src.audio_pcm import bytes_to_pcm_s16le_mono_16k
from src.settings import NVIDIA_ASR_FUNCTION_ID, NVIDIA_ASR_GRPC_URI, NVIDIA_ASR_MODEL

# ~250 ms of 16 kHz mono s16le per chunk
_CHUNK_BYTES = 8000


def _pcm_chunks(pcm: bytes) -> Iterator[bytes]:
    for i in range(0, len(pcm), _CHUNK_BYTES):
        yield pcm[i : i + _CHUNK_BYTES]


def _collect_streaming_transcripts(asr: Any, pcm: bytes, streaming_config: Any) -> str:
    finals: list[str] = []
    last_partial = ""
    responses = asr.streaming_response_generator(_pcm_chunks(pcm), streaming_config)
    for response in responses:
        for result in response.results:
            if not result.alternatives:
                continue
            text = (result.alternatives[0].transcript or "").strip()
            if not text:
                continue
            if result.is_final:
                finals.append(text)
            else:
                last_partial = text
    joined = " ".join(finals).strip()
    if joined:
        return joined
    return last_partial.strip()


def transcribe_nvidia_bytes(
    data: bytes,
    *,
    filename_hint: str = "audio.webm",
    api_key: str | None = None,
) -> str:
    """Transcribe using NVIDIA Cloud NIM (nemotron-asr-streaming) over Riva gRPC."""
    try:
        from riva.client import ASRService, Auth, RecognitionConfig, StreamingRecognitionConfig
        from riva.client.proto.riva_audio_pb2 import AudioEncoding
    except ImportError as e:
        raise ValueError(
            "Nemotron ASR requires the Riva client. Install: pip install nvidia-riva-client grpcio",
        ) from e

    _ = filename_hint  # reserved for future format hints
    from src.settings import NVIDIA_API_KEY as _NV_DEFAULT

    key = (api_key or _NV_DEFAULT or "").strip()
    if not key:
        raise ValueError(
            "NVIDIA API key missing for speech-to-text. Set NVIDIA_API_KEY in .env or paste it under NVIDIA (plan + voice STT).",
        )

    fid = NVIDIA_ASR_FUNCTION_ID.strip()
    if not fid:
        raise ValueError("NVIDIA_ASR_FUNCTION_ID is not set.")

    pcm = bytes_to_pcm_s16le_mono_16k(data)

    auth = Auth(
        ssl_root_cert=None,
        use_ssl=True,
        uri=NVIDIA_ASR_GRPC_URI.strip() or "grpc.nvcf.nvidia.com:443",
        metadata_args=[
            ["function-id", fid],
            ["authorization", f"Bearer {key}"],
        ],
    )
    asr = ASRService(auth)

    model = (NVIDIA_ASR_MODEL or "").strip()
    inner = RecognitionConfig(
        encoding=AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=16000,
        audio_channel_count=1,
        language_code="en-US",
        max_alternatives=1,
        enable_automatic_punctuation=True,
        verbatim_transcripts=False,
    )
    if model:
        inner.model = model

    streaming_config = StreamingRecognitionConfig(config=inner, interim_results=True)
    return _collect_streaming_transcripts(asr, pcm, streaming_config)
