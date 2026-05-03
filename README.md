# Technical Interview Agent

AI-powered structured technical interviews using **Groq**. Input a job description and candidate profile; get a tailored question plan, a real-time conversational interview, and a structured evaluation with transcript citations.

## Quick start

```bash
cd "Technical Interview Agent"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# Edit .env — set GROQ_API_KEY
uvicorn web_app:app --host 127.0.0.1 --port 8840
```

Open **http://127.0.0.1:8840**

## Requirements coverage

| Requirement | Where |
|-------------|--------|
| JD + profile → tailored questions | `src/question_generator.py` |
| Real-time conversational interview | `src/conductor.py` + streaming `/api/chat/stream` |
| Structured evaluation + quotes | `src/evaluator.py` + `samples/evaluation_sample.json` |
| Separable components | `src/schemas.py` + module boundaries |
| Silence / off-topic / clarify | `src/policies.py` + conductor prompts |
| No coaching mid-interview | Conductor system rules; clarifications only |

## Demo roles

Two fixture pairs live under `fixtures/` (backend platform engineer, ML engineer).

## License

MIT
