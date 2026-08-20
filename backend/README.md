# XYZ AI backend

FastAPI service with secure mock school tools, role verification, conversational state, escalation workflow, optional local Ollama response polishing, and CSV-backed school data.

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python scripts/generate_school_csv.py
uvicorn app.main:app --reload --port 8000
```

To enable local natural-language phrasing and tool selection, install/run Ollama and set `OLLAMA_MODEL` in `.env`, e.g. `ollama pull llama3.2` then `ollama serve`. The deterministic tool layer works without Ollama. The app loads this `.env` at startup.

Run `python scripts/generate_school_csv.py` to create the ignored local demo data. It generates `data/students.csv` with 1,200 mock student records (Classes 1–10, Sections A–C, 40 students per section), `data/users.csv` with matching mock portal accounts, and date-level attendance records. Running it again overwrites attendance edits saved by teachers.

At `LOG_LEVEL=INFO`, the backend console logs `OLLAMA_PLAN_REQUEST` / `OLLAMA_PLAN_OUTPUT` and `OLLAMA_RESPONSE_REWRITE_REQUEST` / `OLLAMA_RESPONSE_REWRITE_OUTPUT`. These log the exact credential-free JSON exchanged with Ollama to make model decisions easy to inspect during development.
