"""User-visible messages for NVIDIA NIM / OpenAI client errors (rate limits, TPM, etc.)."""

from __future__ import annotations

_RATE_HINT = (
    "Rate limit or quota hit on the inference API. Wait and retry, or adjust limits in your NVIDIA / NGC billing. "
    "Shorten inputs or reduce max_tokens if you see TPM or request-size errors."
)

_PAYLOAD_HINT = (
    "Request too large for the current token-per-minute (TPM) or max-size policy. "
    "Shorten job description, candidate profile, or transcript context; tune PLAN_MAX_* env vars."
)


def is_api_payload_too_large(exc: BaseException) -> bool:
    try:
        from openai import APIStatusError

        if isinstance(exc, APIStatusError) and getattr(exc, "status_code", None) == 413:
            return True
    except ImportError:
        pass
    msg = str(exc).lower()
    if "error code: 413" in msg or " status code: 413" in msg:
        return True
    if "request too large" in msg:
        return True
    if "tokens per minute" in msg or "tokens per min" in msg or " tpm" in msg:
        return True
    return False


def is_api_rate_limit(exc: BaseException) -> bool:
    if is_api_payload_too_large(exc):
        return False
    try:
        from openai import APIStatusError

        if isinstance(exc, APIStatusError) and getattr(exc, "status_code", None) == 429:
            return True
    except ImportError:
        pass
    msg = str(exc).lower()
    if "error code: 429" in msg or " status code: 429" in msg:
        return True
    if "rate_limit" in msg or "rate limit" in msg:
        return True
    return False


def api_http_detail(exc: BaseException, *, context: str = "API") -> str:
    if is_api_payload_too_large(exc):
        return f"{context}: {_PAYLOAD_HINT} Raw: {exc!s}"
    if is_api_rate_limit(exc):
        return f"{context}: {_RATE_HINT} Raw: {exc!s}"
    return f"{context} error: {exc!s}"


def api_ws_detail(exc: BaseException, *, max_len: int = 780) -> str:
    if is_api_payload_too_large(exc):
        s = f"{_PAYLOAD_HINT} ({exc!s})"
    elif is_api_rate_limit(exc):
        s = f"{_RATE_HINT} ({exc!s})"
    else:
        s = str(exc)
    if len(s) > max_len:
        return s[: max_len - 1] + "…"
    return s
