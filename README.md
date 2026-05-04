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
3. After each interviewer line (browser speech), it’s your turn: click **Start answer**, speak (everything until **End & send** is recorded and transcribed). **ffmpeg** on `PATH` is required to decode WebM from the mic for Nemotron ASR.
4. **Next question** (toolbar or voice panel) **stops** the interviewer’s speech and moves to the next planned question. If you never start an answer, **silence nudge / auto-advance** still applies (see `.env`: `VOICE_SILENCE_NUDGE_SECONDS`, `VOICE_SILENCE_SKIP_SECONDS`).
5. **Evaluate** builds a **performance report** (strengths, gaps, scores) from the **full cumulative transcript** across all questions (see evaluator prompt).

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

## Run metadata (models, duration, tech stack)

**Repository:** [github.com/Aivar-sanjeev/Technical-Interview-Agent](https://github.com/Aivar-sanjeev/Technical-Interview-Agent)

### Project duration (`main`: first commit → latest)

| | |
|---|---|
| **First commit** (root of `main`) | `7d7270c` — **2026-05-03 21:38:39 UTC** |
| **Latest commit** (`main` tip) | `f8b46e7` — **2026-05-04 06:28:27 UTC** |
| **Elapsed (root → tip)** | **8 h 49 m 48 s** (31,788 s) across **12** commits on `main` |

Recompute (PowerShell, repo root):

```powershell
$root = git rev-list --max-parents=0 main; $head = git rev-parse main
$t0 = [int](git show -s --format=%at $root); $t1 = [int](git show -s --format=%at $head)
$ts = [TimeSpan]::FromSeconds($t1 - $t0)
"Elapsed: {0}d {1}h {2}m {3}s  (commits: {4})" -f $ts.Days, $ts.Hours, $ts.Minutes, $ts.Seconds, (git rev-list --count main)
```

Use the table for each **end-to-end interview session** (plan → voice/text session → evaluate). Fill timing from wall clock or server logs.

### Tech stack

Python 3, **FastAPI**, **uvicorn**, **SSE**, **NVIDIA NIM** (`integrate.api.nvidia.com`), **Nemotron ASR** (Riva gRPC), browser **Web Speech API** (TTS), optional **ffmpeg** for WebM decode.

### Models used (configure in `.env` / UI)

| Role | Typical model / service | Notes |
|------|-------------------------|--------|
| Plan + interview + evaluation | `openai/gpt-oss-20b` (default in app) | NVIDIA NIM chat; see `.env.example`. |
| Candidate speech | Nemotron ASR (streaming) | Riva / NVIDIA STT endpoint. |

### Run log (fill per execution)

| Task / run | Command / flow | Models used | Tech stack | Started (UTC) | Finished (UTC) | Wall duration |
|------------|----------------|-------------|------------|---------------|----------------|-----------------|
| Example full interview | `uvicorn web_app:app …` → plan → session → Evaluate | as configured | stack above | — | — | — |

## License

MIT
