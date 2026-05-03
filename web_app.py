"""
Technical Interview Agent — FastAPI UI and APIs.

Run: uvicorn web_app:app --host 127.0.0.1 --port 8840
"""

from __future__ import annotations

import json
import secrets
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.conductor import stream_interviewer_reply
from src.evaluator import evaluate_with_repair, validate_report_against_transcript
from src.question_generator import generate_interview_plan
from src.schemas import InterviewPlan, Transcript, TranscriptTurn

app = FastAPI(title="Technical Interview Agent", version="0.3.0")
app.mount("/fixtures", StaticFiles(directory=str(_ROOT / "fixtures")), name="fixtures")

SESSIONS: dict[str, dict[str, Any]] = {}
_SESSION_LOCK = threading.Lock()


def _tid() -> str:
    return str(uuid.uuid4())


def _html_path() -> Path:
    return _ROOT / "web" / "index.html"


def _transcript_from_session(turns: list[dict[str, Any]]) -> Transcript:
    clean: list[TranscriptTurn] = []
    for t in turns:
        clean.append(TranscriptTurn.model_validate(t))
    return Transcript(turns=clean)


class PlanIn(BaseModel):
    job_description: str = Field(min_length=40)
    candidate_profile: str = Field(min_length=40)
    groq_api_key: str | None = Field(default=None)


class SessionStartIn(BaseModel):
    plan: dict[str, Any]
    groq_api_key: str | None = None


class EvaluateIn(BaseModel):
    session_id: str = Field(min_length=8)
    job_description: str = Field(min_length=40)
    candidate_profile: str = Field(min_length=40)
    groq_api_key: str | None = None


class SessionChatIn(BaseModel):
    session_id: str = Field(min_length=8)
    message: str = ""
    event_type: str = Field(default="normal")
    action: str = Field(default="reply")

    @field_validator("event_type")
    @classmethod
    def _ev(cls, v: str) -> str:
        allowed = {"normal", "silence", "clarify_request", "off_topic"}
        if v not in allowed:
            raise ValueError(f"event_type must be one of {allowed}")
        return v

    @field_validator("action")
    @classmethod
    def _act(cls, v: str) -> str:
        allowed = {"opening", "reply", "advance_question"}
        if v not in allowed:
            raise ValueError(f"action must be one of {allowed}")
        return v


def _get_session(session_id: str) -> dict[str, Any]:
    with _SESSION_LOCK:
        s = SESSIONS.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    return s


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    path = _html_path()
    if not path.is_file():
        return "<p>Missing web/index.html</p>"
    return path.read_text(encoding="utf-8")


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


@app.post("/api/session/start")
async def session_start(body: SessionStartIn) -> JSONResponse:
    try:
        plan = InterviewPlan.model_validate(body.plan)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid plan: {e!s}") from e
    sid = secrets.token_urlsafe(24)
    with _SESSION_LOCK:
        SESSIONS[sid] = {
            "plan": plan,
            "turns": [],
            "q_index": 0,
            "groq_api_key": (body.groq_api_key or "").strip() or None,
            "followups": {},
        }
    return JSONResponse({"session_id": sid, "question_count": len(plan.questions)})


@app.post("/api/session/advance")
async def session_advance(body: dict[str, str]) -> JSONResponse:
    sid = (body.get("session_id") or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="session_id required")
    s = _get_session(sid)
    plan: InterviewPlan = s["plan"]
    with _SESSION_LOCK:
        if s["q_index"] + 1 >= len(plan.questions):
            return JSONResponse({"ok": False, "detail": "Already at last question"})
        s["q_index"] += 1
        s["turns"].append(
            {
                "id": _tid(),
                "role": "system",
                "text": "Interviewer moved to the next planned question.",
                "ts": time.time(),
            }
        )
        q_index = s["q_index"]
    return JSONResponse({"ok": True, "q_index": q_index, "question_id": plan.questions[q_index].id})


@app.post("/api/chat/stream")
async def chat_stream(body: SessionChatIn) -> StreamingResponse:
    s = _get_session(body.session_id)
    api_key = s.get("groq_api_key")

    def gen():
        event_type = body.event_type
        interaction = "continue"
        if body.action in ("opening", "advance_question"):
            interaction = "opening"

        try:
            if body.action == "advance_question":
                with _SESSION_LOCK:
                    plan_live: InterviewPlan = s["plan"]
                    if s["q_index"] + 1 >= len(plan_live.questions):
                        yield f"data: {json.dumps({'error': 'no_more_questions'})}\n\n".encode("utf-8")
                        return
                    s["q_index"] += 1
                    s["turns"].append(
                        {
                            "id": _tid(),
                            "role": "system",
                            "text": "Interviewer moved to the next planned question.",
                            "ts": time.time(),
                        }
                    )

            if body.action == "reply":
                if event_type == "silence":
                    with _SESSION_LOCK:
                        qid_live = s["plan"].questions[s["q_index"]].id
                        if body.message.strip():
                            s["turns"].append(
                                {
                                    "id": _tid(),
                                    "role": "candidate",
                                    "text": body.message.strip(),
                                    "ts": time.time(),
                                }
                            )
                        else:
                            s["turns"].append(
                                {
                                    "id": _tid(),
                                    "role": "system",
                                    "text": "Interview note: candidate pause / silence for this turn.",
                                    "ts": time.time(),
                                }
                            )
                        s["followups"][qid_live] = int(s["followups"].get(qid_live, 0)) + 1
                elif body.message.strip():
                    cand = {
                        "id": _tid(),
                        "role": "candidate",
                        "text": body.message.strip(),
                        "ts": time.time(),
                    }
                    with _SESSION_LOCK:
                        s["turns"].append(cand)
                        qid_live = s["plan"].questions[s["q_index"]].id
                        s["followups"][qid_live] = int(s["followups"].get(qid_live, 0)) + 1
                elif event_type in ("clarify_request", "off_topic"):
                    pass
                else:
                    yield f"data: {json.dumps({'error': 'empty_message'})}\n\n".encode("utf-8")
                    return

            with _SESSION_LOCK:
                plan_obj: InterviewPlan = s["plan"]
                q_index_live = int(s["q_index"])
                fq_after = dict(s["followups"])
                tr2 = _transcript_from_session(list(s["turns"]))

            if q_index_live >= len(plan_obj.questions):
                yield f"data: {json.dumps({'error': 'no_more_questions'})}\n\n".encode("utf-8")
                return

            qid2 = plan_obj.questions[q_index_live].id
            follow_count2 = int(fq_after.get(qid2, 0))

            parts: list[str] = []
            for token in stream_interviewer_reply(
                plan_obj,
                tr2,
                q_index_live,
                candidate_message=body.message.strip(),
                event_type=event_type,
                followups_on_current=follow_count2,
                interaction=interaction,
                api_key=api_key,
            ):
                parts.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n".encode("utf-8")

            full = "".join(parts).strip()
            if full:
                turn = {
                    "id": _tid(),
                    "role": "interviewer",
                    "text": full,
                    "ts": time.time(),
                }
                with _SESSION_LOCK:
                    s["turns"].append(turn)
            yield f"data: {json.dumps({'done': True})}\n\n".encode("utf-8")
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n".encode("utf-8")

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/evaluate")
async def api_evaluate(body: EvaluateIn) -> JSONResponse:
    s = _get_session(body.session_id)
    plan: InterviewPlan = s["plan"]
    transcript = _transcript_from_session(list(s["turns"]))
    key = (body.groq_api_key or "").strip() or s.get("groq_api_key")

    if not transcript.turns:
        raise HTTPException(status_code=400, detail="Transcript is empty.")

    try:
        report, issues = evaluate_with_repair(
            job_description=body.job_description,
            candidate_profile=body.candidate_profile,
            plan=plan,
            transcript=transcript,
            api_key=key,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Groq error: {e!s}") from e

    residual = validate_report_against_transcript(report, transcript)
    return JSONResponse(
        {
            "report": report.model_dump(),
            "validation_issues": issues,
            "residual_issues": residual,
        }
    )


@app.get("/api/session/{session_id}")
async def session_get(session_id: str) -> JSONResponse:
    s = _get_session(session_id)
    plan: InterviewPlan = s["plan"]
    return JSONResponse(
        {
            "q_index": s["q_index"],
            "question_id": plan.questions[s["q_index"]].id if s["q_index"] < len(plan.questions) else None,
            "turns": s["turns"],
            "plan": plan.model_dump(),
        }
    )
