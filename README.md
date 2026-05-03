# Technical Interview Agent

End-to-end **AI technical interviews** on **NVIDIA NIM** (`integrate.api.nvidia.com`): **openai/gpt-oss-20b** for plans, live interview, and evaluation; **Nemotron ASR streaming** (Riva gRPC) for candidate speech; interviewer audio uses the browser’s **Speech Synthesis** API (no cloud TTS key required).

**Repository:** https://github.com/Aivar-sanjeev/Technical-Interview-Agent  

**Architecture / separation of concerns:** see **[ARCHITECTURE.md](ARCHITECTURE.md)**.

## Quick start

```powershell
cd "Technical Interview Agent"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# Set NVIDIA_API_KEY in .env (https://build.nvidia.com)
uvicorn web_app:app --host 127.0.0.1 --port 8840
```

Open **http://127.0.0.1:8840**

Paste your **NVIDIA API key** in the UI (optional if set in `.env`); it is sent only to your local server and then to NVIDIA — do not expose this app publicly without authentication.

### Voice + face interview (Nemotron STT + browser TTS)

1. On the plan screen, choose **Start as: Voice + face** — the voice WebSocket connects automatically after you start the session (camera + mic permission).
2. **Full-screen** layout focuses on you and the interviewer orb; use **Exit fullscreen** to show the normal page chrome again (session stays live).
3. After each interviewer line (browser speech), the app **listens automatically** and stops when you pause (VAD-style); you can still **Hold to speak (override)** if needed. **ffmpeg** on `PATH` is required to decode WebM from the mic for Nemotron ASR.
4. If you stay silent, the server sends a **short nudge**, then **auto-advances** to the next planned question (timings: `VOICE_SILENCE_NUDGE_SECONDS`, `VOICE_SILENCE_SKIP_SECONDS` in `.env`).
5. **Evaluate** uses the **full cumulative transcript** across all questions (see evaluator prompt).

### Voice pacing (wait longer before sending your answer)

The browser **auto-listen** path waits for real speech, then for a **longer pause** (~1.2s of quiet at 60fps) before it stops recording and sends audio to Nemotron. It also waits at least **~2.8s** after “Your turn” before it can end the clip on silence alone, so brief thinking pauses do not cut you off. The hard cap per listen is **90s** (then it sends whatever was captured).

To tune **server-side** silence (nudge → auto next question), edit `.env`:

- **`VOICE_SILENCE_NUDGE_SECONDS`** (default **22**) — how long after the interviewer finishes before a silence nudge.
- **`VOICE_SILENCE_SKIP_SECONDS`** (default **28**) — additional wait after the nudge before moving to the next planned question if you still have not replied.

Increase those values if you want more time before auto-advance. **Next question** in the UI still advances manually when you are ready.

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
