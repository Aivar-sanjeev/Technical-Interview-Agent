# Technical Interview Agent

End-to-end **AI technical interviews** powered by **Groq**: tailored question plans from a job description and candidate profile, a **streaming** conversational interview with adaptive follow-ups, and a **structured evaluation** that cites what the candidate actually said.

**Repository:** https://github.com/Aivar-sanjeev/Technical-Interview-Agent  

**Architecture / separation of concerns:** see **[ARCHITECTURE.md](ARCHITECTURE.md)**.

## Quick start

```powershell
cd "Technical Interview Agent"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# Set GROQ_API_KEY in .env (https://console.groq.com)
uvicorn web_app:app --host 127.0.0.1 --port 8840
```

Open **http://127.0.0.1:8840**

You can also paste a Groq key in the UI (optional); it is sent only to your local server and then to Groq — do not expose this app publicly without authentication.

### Voice + face interview (Groq Whisper + Orpheus)

1. On the plan screen, choose **Start as: Voice + face** (voice-first skips the text SSE opening).
2. After the session starts, allow **camera + microphone**, then **Connect voice**.
3. **Hold to speak** — release to send audio; Groq **Whisper** transcribes it, **Llama** generates the interviewer reply, **Orpheus** reads it aloud (WAV chunks over WebSocket).
4. **Next (voice)** advances the planned question on the voice lane. Text controls (Send / Next) still work if you switch back mentally — hide the text box when using voice-only flow.

Requires **HTTPS or localhost** for `getUserMedia` in most browsers.

## Two demo roles (fixtures)

| Role | Fixture files |
|------|----------------|
| Senior backend / platform | `fixtures/jd_backend.md` + `fixtures/profile_backend.md` |
| ML engineer (ranking & retrieval) | `fixtures/jd_ml.md` + `fixtures/profile_ml.md` |

In the UI, use **Load backend fixture** or **Load ML fixture**, then **Generate question plan**.

## API highlights

| Endpoint | Purpose |
|----------|---------|
| `POST /api/plan` | JD + profile → `InterviewPlan` |
| `POST /api/session/start` | Start in-memory session from a plan |
| `POST /api/chat/stream` | SSE stream for interviewer (`action`: `opening`, `reply`, `advance_question`; `event_type`: `normal`, `silence`, `clarify_request`, `off_topic`) |
| `POST /api/evaluate` | JD + profile + session → structured `EvaluationReport` + validation notes |

## Sample evaluation artifact

Example JSON shape (synthetic evidence ids): **[samples/evaluation_sample.json](samples/evaluation_sample.json)**  
Also served statically at **http://127.0.0.1:8840/samples/evaluation_sample.json** when the app is running.

## Requirements checklist

| Requirement | Implementation |
|-------------|----------------|
| JD + profile → tailored questions | `src/question_generator.py` |
| Real-time conversational interview | `src/conductor.py` + SSE in `web_app.py` |
| Structured report + citations | `src/evaluator.py` + `validate_report_against_transcript` |
| Separable components | `src/schemas.py` contracts; see ARCHITECTURE.md |
| Silence / off-topic / clarify | `src/policies.py` + `event_type` in API |
| No coaching mid-interview | Conductor system prompt; clarifications only |

## License

MIT
