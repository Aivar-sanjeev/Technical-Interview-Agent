"""
Technical Interview Agent — FastAPI entrypoint.

Phase 1: plan generation API only (interview + evaluation arrive in later phases).
Run: uvicorn web_app:app --host 127.0.0.1 --port 8840
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from src.question_generator import generate_interview_plan

app = FastAPI(title="Technical Interview Agent", version="0.1.0")


class PlanIn(BaseModel):
    job_description: str = Field(min_length=40)
    candidate_profile: str = Field(min_length=40)
    groq_api_key: str | None = Field(default=None, description="Optional override; else GROQ_API_KEY from .env")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    path = _ROOT / "web" / "index.html"
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return "<p>Phase 1: POST JSON to <code>/api/plan</code> with job_description and candidate_profile.</p>"


@app.post("/api/plan")
async def api_plan(body: PlanIn) -> JSONResponse:
    try:
        plan = generate_interview_plan(
            body.job_description,
            body.candidate_profile,
            api_key=body.groq_api_key,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Groq error: {e!s}") from e
    return JSONResponse(plan.model_dump())
