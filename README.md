# XYZ AI — Human-like School Assistant

XYZ AI is a standalone Applied AI school-assistant demo. It supports chat, browser voice, a responsive avatar, four secure role personas, multilingual selection, conversation context, mock SIS actions, confirmed human escalation, and a staff-only attendance dashboard.

## Repository layout

```
xyz-ai/
├── frontend/       # Next.js, React, Tailwind, voice and avatar UI
└── backend/        # FastAPI, secure mock services, optional Ollama client
```

## Run locally

1. Start the backend using [backend/README.md](backend/README.md).
2. Start the frontend using [frontend/README.md](frontend/README.md).
3. Visit `http://localhost:3000`, choose a demo portal role, and ask its example question.

The UI obtains a signed demo session from the backend rather than trusting a role typed into chat. The backend enforces student/parent record ownership, teacher class assignment, and principal-only analytics at the mock API layer. Ollama is optional: set `OLLAMA_MODEL` after installing Ollama locally to make approved factual answers more conversational, while authorization and actions remain deterministic and server-side.

## Demo scenarios

| Demo role | Try this |
| --- | --- |
| Student | `What is my attendance?` |
| Parent | `How much attendance does my child have?` then `recent attendance` |
| Teacher | `Mark Rahul absent today.` |
| Principal | `What is the overall attendance?` |

For escalation, as a parent say `I want to talk to my child's teacher`, then confirm with `yes`. The confirmation is only reported after the mock support service succeeds.

## Single-page portal and CSV data

The root page (`/`) is a single shared portal: every authenticated Student, Parent, Teacher, and Principal can use the visible XYZ AI chatbot. The attendance workspace is conditionally rendered on that same page for Teacher and Principal accounts only, and is also restricted at the API layer.

Run `python backend/scripts/generate_school_csv.py` to create the ignored local mock SIS files. It generates `students.csv` with **Classes 1–10**, **sections A–C**, and **40 students in each section** (1,200 students total), plus matching mock portal accounts and date-level attendance records. Each of the 30 sections has a distinct assigned teacher.

| Role | Demo login | Access |
| --- | --- | --- |
| Student | `student.rahul@xyzschool.edu` / `student123` | Chat only |
| Parent | `parent.priya@xyzschool.edu` / `parent123` | Chat only |
| Teacher | `teacher.8a@xyzschool.edu` / `teacher8a123` | Chat plus can view and mark only assigned Class 8A |
| Principal | `principal.mehta@xyzschool.edu` / `principal123` | Can review all 30 class sections and school metrics |

Student and parent tokens are rejected from every attendance-dashboard endpoint, while principal tokens are rejected from attendance-edit actions. The supplied credentials are for local demonstration only; connect the login endpoint to the school’s SSO provider in production.

## Ollama action flow

With Ollama running locally, the backend sends each chat request to the configured local model (`llama3.2:latest`) with a restricted tool list. The model can select an attendance lookup, three-month attendance lookup, dated teacher attendance mark, or principal analytics action. For example: `Mark Rahul absent on 2026-08-15`. FastAPI resolves names against the CSV, validates the calendar date and signed role/class claims, executes the approved action, and persists teacher updates—so the model never receives raw credentials or directly edits data.
