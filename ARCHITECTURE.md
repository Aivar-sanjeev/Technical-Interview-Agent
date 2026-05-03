# Architecture: separating plan, conduct, and evaluate

This project satisfies the requirement that **question generation**, **interview conduct**, and **evaluation** are separable: each has its own module, prompts, and Groq model settings. They communicate only through **shared contracts** (`src/schemas.py`), not by calling each other’s internals.

## Component map

| Component | Module | Responsibility | Inputs | Outputs |
|-----------|--------|----------------|--------|---------|
| **Plan** | `src/question_generator.py` | Build a tailored `InterviewPlan` from JD + profile | `job_description`, `candidate_profile` | `InterviewPlan` (JSON-serializable) |
| **Conduct** | `src/conductor.py` + `src/policies.py` | Stream the next interviewer turn; enforce interview behavior | `InterviewPlan`, `Transcript`, `q_index`, candidate text, `event_type`, `interaction` | Token stream → full interviewer message (persisted in `web_app` session) |
| **Evaluate** | `src/evaluator.py` | Produce `EvaluationReport` after the session | JD, profile, `InterviewPlan`, `Transcript` | `EvaluationReport` + optional validation issues |

Supporting pieces:

- **`src/schemas.py`** — Pydantic models used by all three layers (`InterviewPlan`, `Transcript`, `EvaluationReport`, etc.). Changing evaluation shape does not force changes to the conductor unless you intentionally extend the API contract.
- **`src/config.py`** — API key and **separate model env vars** (`GROQ_MODEL_PLAN`, `GROQ_MODEL_INTERVIEW`, `GROQ_MODEL_EVAL`) so you can swap models per phase without code changes.
- **`web_app.py`** — HTTP orchestration only: sessions, SSE streaming, and calling into the three modules. It is not the “business logic” for scoring or questioning.

## Why this separation matters

1. **Swap or retrain one concern** — e.g. replace `evaluator.py` with a rules engine + LLM hybrid without touching `question_generator.py`.
2. **Test policies without Groq** — `policies.py` is pure string context; unit tests can assert prefixes for silence / clarify / off-topic without API calls.
3. **Avoid coaching leakage** — the conductor system prompt is isolated from the evaluator; the evaluator never runs mid-interview.
4. **Evidence-grounded reports** — `evaluator.validate_report_against_transcript` checks that quotes appear in candidate text; a repair pass can ask the model to fix only evidence fields.

## Data flow (typical session)

1. Client POST `/api/plan` → `generate_interview_plan` → `InterviewPlan`.
2. Client POST `/api/session/start` with that plan → in-memory session (`plan`, `turns`, `q_index`).
3. Client POST `/api/chat/stream` (SSE) → append candidate/system turns as needed → `stream_interviewer_reply` → append interviewer turn.
4. Client POST `/api/evaluate` with JD/profile + `session_id` → `evaluate_with_repair` → structured JSON + validation notes.

## Real-time transport

The conductor uses Groq **streaming** completions. `web_app` forwards deltas as **SSE** (`text/event-stream`) so the browser can render tokens incrementally.

## Scenario handling (no mid-interview coaching)

- **Silence / stuck** — `event_type=silence` plus optional note; `policies.py` injects interviewer-only guidance (no solutions).
- **Clarify** — `event_type=clarify_request`; policy restricts to rephrasing / scoping.
- **Off-topic** — `event_type=off_topic`; policy restricts to polite redirect.

The LLM still composes natural language, but deterministic prefixes keep behavior reviewable and consistent.

## Sample artifact

See `samples/evaluation_sample.json` for an example of the structured report shape (with synthetic evidence turn ids for illustration).
