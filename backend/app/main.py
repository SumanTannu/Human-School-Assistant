import csv
import json
import logging
import os
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import httpx
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("xyz_ai.ollama")

ROLE_VOICE_PROMPTS = {
    "student": "Sound warm, encouraging, age-appropriate, and academically supportive. Explain clearly without being patronizing.",
    "parent": "Sound caring, patient, reassuring, and concise. Be clear about attendance facts and practical next steps.",
    "teacher": "Sound professional, efficient, and precise. Confirm administrative changes clearly and avoid unnecessary filler.",
    "principal": "Sound professional, concise, and insight-oriented. Emphasize trends, context, and decisions.",
}

RESPONSE_MODE_PROMPTS = {
    "greeting": "Give a warm, natural greeting in the user's language. Introduce yourself briefly and invite a school-related request without sounding scripted.",
    "attendance": "Share the approved attendance facts in a friendly, natural sentence. Be helpful but do not sound like a report or add unnecessary praise.",
    "history": "For a single date, the approved record is available. Reply as a warm, human assistant in one short sentence, for example: 'Yes, you were present on 17 August 2026.' State the approved Present or Absent status and date naturally. Never say that information is unavailable, and do not mention overall percentages, other dates, ranges, or advice. For a range, answer the requested overall attendance for that period first, then optionally state the period; do not substitute a single-day status or invent causes.",
    "attendance_update": "Confirm the approved attendance change in a clear, natural sentence. Sound like a helpful colleague, not an audit log.",
    "analytics": "Explain approved school metrics naturally and clearly, with one useful management takeaway. Avoid report-like headings or rigid phrasing.",
    "escalation": "Explain the approved escalation state with empathy and clear next steps. Never claim a request is submitted until the facts say it was submitted.",
    "clarification": "Ask one short, natural question for the missing information needed to proceed.",
    "safety": "Set the boundary politely in natural language, then offer the closest permitted school assistance.",
    "general": "Answer naturally as a thoughtful human school assistant using only the approved facts.",
}

app = FastAPI(title="XYZ AI", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JWT_SECRET = os.getenv("JWT_SECRET", "development-only-change-me")
JWT_ALGORITHM = "HS256"
bearer = HTTPBearer()

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
STUDENTS_CSV = DATA_DIR / "students.csv"
USERS_CSV = DATA_DIR / "users.csv"
ATTENDANCE_RECORDS_CSV = DATA_DIR / "attendance_records.csv"


def load_students() -> dict[str, dict]:
    """Load the 1,200 mock SIS records from CSV, not application source code."""
    students: dict[str, dict] = {}
    with STUDENTS_CSV.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            students[row["student_id"]] = {
                "name": row["name"], "grade": int(row["grade"]), "section": row["section"],
                "class_id": f"grade_{row['grade']}{row['section'].lower()}", "teacher_id": row["teacher_id"], "teacher_name": row["teacher_name"],
                "attendance": {"present": int(row["present_days"]), "total": int(row["total_days"])},
                "monthly": {"June": float(row["june_attendance"]), "July": float(row["july_attendance"]), "August": float(row["august_attendance"])},
            }
    return students


def save_students() -> None:
    """Persist teacher attendance edits back to the CSV mock SIS data."""
    fields = ["student_id", "name", "grade", "section", "teacher_id", "teacher_name", "present_days", "total_days", "june_attendance", "july_attendance", "august_attendance"]
    with STUDENTS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for student_id, student in STUDENTS.items():
            writer.writerow({"student_id": student_id, "name": student["name"], "grade": student["grade"], "section": student["section"], "teacher_id": student["teacher_id"], "teacher_name": student["teacher_name"], "present_days": student["attendance"]["present"], "total_days": student["attendance"]["total"], "june_attendance": student["monthly"]["June"], "july_attendance": student["monthly"]["July"], "august_attendance": student["monthly"]["August"]})


def load_attendance_records() -> dict[tuple[str, str], str]:
    with ATTENDANCE_RECORDS_CSV.open(encoding="utf-8", newline="") as handle:
        return {(row["date"], row["student_id"]): row["status"] for row in csv.DictReader(handle)}


def save_attendance_records() -> None:
    with ATTENDANCE_RECORDS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "student_id", "status"])
        writer.writeheader()
        for (attendance_date, student_id), attendance_status in sorted(ATTENDANCE_RECORDS.items()):
            writer.writerow({"date": attendance_date, "student_id": student_id, "status": attendance_status})


def load_accounts() -> dict[str, dict]:
    accounts: dict[str, dict] = {}
    with USERS_CSV.open(encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            accounts[row["email"].lower()] = {"id": f"user_{index}", "name": row["name"], "role": row["role"], "student_id": row["student_id"] or None, "child_ids": [item for item in row["child_ids"].split("|") if item], "class_ids": [item for item in row["class_ids"].split("|") if item], "password": row["password"]}
    return accounts


ACCOUNTS = load_accounts()
STUDENTS = load_students()
ATTENDANCE_RECORDS = load_attendance_records()
ATTENDANCE_DATES = sorted({record_date for record_date, _ in ATTENDANCE_RECORDS})
CONVERSATIONS: dict[str, dict] = {}


class PortalLoginRequest(BaseModel):
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=1, max_length=128)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(min_length=3, max_length=100)
    language: str = "English"


class ChatResponse(BaseModel):
    response: str
    intent: str
    action_taken: bool = False
    escalation_options: list[str] = []
    source: Literal["rules", "ollama"] = "rules"


class MarkAttendanceRequest(BaseModel):
    student_id: str
    status: Literal["present", "absent"]
    attendance_date: date


def create_token(account: dict) -> str:
    payload = {**account, "exp": datetime.now(timezone.utc) + timedelta(hours=8)}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer)) -> dict:
    try:
        return jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Your secure session has expired. Please sign in again.")


def require_role(user: dict, *roles: str) -> None:
    if user["role"] not in roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Your account is not permitted to perform that action.")


def attendance_for(user: dict, student_id: str) -> dict:
    """Authorization belongs beside the data/tool, never solely in the model prompt."""
    if user["role"] == "student" and user.get("student_id") != student_id:
        raise HTTPException(status_code=403, detail="Students can only view their own attendance.")
    if user["role"] == "parent" and student_id not in user.get("child_ids", []):
        raise HTTPException(status_code=403, detail="Parents can only view attendance for linked children.")
    if user["role"] not in ("student", "parent", "teacher", "principal"):
        raise HTTPException(status_code=403, detail="Access denied.")
    return STUDENTS[student_id]


def school_analytics(user: dict) -> str:
    require_role(user, "principal")
    return "School-wide attendance is 92.8% this term, up 1.4 percentage points from last month. Grade 8A is at 91.2%."


def mark_attendance(user: dict, student_id: str, attendance_status: str, attendance_date: date) -> str:
    require_role(user, "teacher")
    student = STUDENTS.get(student_id)
    if not student or student["class_id"] not in user.get("class_ids", []):
        raise HTTPException(status_code=403, detail="You may only mark attendance for students in your assigned classes.")
    # Mock API mutationâ€”this is where a real SIS integration would be called.
    record = student["attendance"]
    date_key = attendance_date.isoformat()
    if date_key not in ATTENDANCE_DATES:
        raise HTTPException(status_code=422, detail=f"Attendance can be edited only between {ATTENDANCE_DATES[0]} and {ATTENDANCE_DATES[-1]}.")
    previous_status = ATTENDANCE_RECORDS.get((date_key, student_id))
    if previous_status != attendance_status:
        record["present"] += 1 if attendance_status == "present" else -1
    ATTENDANCE_RECORDS[(date_key, student_id)] = attendance_status
    save_students()
    save_attendance_records()
    return f"Marked {student['name']} {attendance_status} for {attendance_date.strftime('%d %b %Y')}."


def attendance_payload(student_id: str, attendance_date: str | None = None) -> dict:
    student = STUDENTS[student_id]
    record = student["attendance"]
    return {
        "student_id": student_id,
        "student_name": student["name"],
        "attendance_percentage": round(record["present"] / record["total"] * 100, 1),
        "present_days": record["present"],
        "total_days": record["total"],
        "monthly": [{"month": month, "percentage": percentage} for month, percentage in student["monthly"].items()],
        "selected_status": ATTENDANCE_RECORDS.get((attendance_date, student_id), "not_marked") if attendance_date else None,
        "recent": [
            {"date": record_date, "status": ATTENDANCE_RECORDS[(record_date, student_id)].title()}
            for record_date in reversed(ATTENDANCE_DATES[-5:])
            if (record_date, student_id) in ATTENDANCE_RECORDS
        ],
    }


def submit_escalation(user: dict, target: str) -> bool:
    if user["role"] not in ("student", "parent"):
        raise HTTPException(status_code=403, detail="Only students and parents can create support requests here.")
    # Mock support API. Only a True result permits the assistant to state confirmation.
    return target in {"teacher", "management"}


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def relative_attendance_window(message: str) -> tuple[str, str] | None:
    """Normalize explicit relative dates against the data calendar before reading CSV records."""
    text = normalized(message)
    dataset_today = datetime.strptime(ATTENDANCE_DATES[-1], "%Y-%m-%d").date()
    if re.search(r"\b(?:this|current)\s+month(?:'s|s)?\b", text):
        start_date = dataset_today.replace(day=1)
        return (start_date.isoformat(), dataset_today.isoformat()) if start_date.isoformat() in ATTENDANCE_DATES else None
    if "last month" in text:
        current_month_start = dataset_today.replace(day=1)
        end_date = current_month_start - timedelta(days=1)
        start_date = end_date.replace(day=1)
        return (start_date.isoformat(), end_date.isoformat()) if start_date.isoformat() in ATTENDANCE_DATES and end_date.isoformat() in ATTENDANCE_DATES else None
    for month_start in (datetime.strptime(record_date, "%Y-%m-%d").date() for record_date in ATTENDANCE_DATES):
        if re.search(r"\b" + re.escape(month_start.strftime("%B").lower()) + r"\b", text):
            dates_in_month = [record_date for record_date in ATTENDANCE_DATES if record_date.startswith(month_start.strftime("%Y-%m"))]
            return (dates_in_month[0], dates_in_month[-1])
    if "last week" in text:
        end_date = dataset_today - timedelta(days=1)
        start_date = end_date - timedelta(days=6)
        return (start_date.isoformat(), end_date.isoformat()) if start_date.isoformat() in ATTENDANCE_DATES and end_date.isoformat() in ATTENDANCE_DATES else None
    period_match = re.search(r"\b(?:last|past)\s+(\d+)\s+(days?|weeks?)\b", text)
    if period_match:
        amount = int(period_match.group(1))
        days = amount * (7 if period_match.group(2).startswith("week") else 1)
        end_date = dataset_today - timedelta(days=1)
        start_date = end_date - timedelta(days=days - 1)
        return (start_date.isoformat(), end_date.isoformat()) if start_date.isoformat() in ATTENDANCE_DATES and end_date.isoformat() in ATTENDANCE_DATES else None
    offset: int | None = None
    if "day before yesterday" in text:
        offset = 2
    elif re.search(r"\byesterday\b", text):
        offset = 1
    elif re.search(r"\btoday\b", text):
        offset = 0
    else:
        match = re.search(r"\b(\d+)\s+days?\s+(?:ago|back)\b", text)
        if match:
            offset = int(match.group(1))
    if offset is None:
        return None
    selected_date = (dataset_today - timedelta(days=offset)).isoformat()
    return (selected_date, selected_date) if selected_date in ATTENDANCE_DATES else None


def requested_language(language: str, text: str) -> str:
    # The selected language is explicit; recognisable scripts provide a graceful fallback.
    if re.search(r"[\u0900-\u097F]", text): return "Hindi"
    if re.search(r"[\u0B80-\u0BFF]", text): return "Tamil"
    if re.search(r"[\u0C00-\u0C7F]", text): return "Telugu"
    if re.search(r"[\u0980-\u09FF]", text): return "Bengali"
    if re.search(r"[\u0600-\u06FF]", text): return "Urdu"
    return language


def is_injection(message: str) -> bool:
    indicators = ("ignore previous", "system prompt", "reveal prompt", "api key", "developer message", "jailbreak")
    return any(term in normalized(message) for term in indicators)


async def improve_with_ollama(facts: str, user: dict, language: str, response_mode: str = "general", required_terms: tuple[str, ...] = ()) -> tuple[str, str]:
    """Optional local model. It receives approved facts only; it has no credentials or tools."""
    model = os.getenv("OLLAMA_MODEL")
    if not model:
        return facts, "rules"
    prompt = (
        f"You are XYZ AI. Reply in {language}. {ROLE_VOICE_PROMPTS[user['role']]} "
        f"Response goal: {RESPONSE_MODE_PROMPTS.get(response_mode, RESPONSE_MODE_PROMPTS['general'])} "
        "Use only the approved facts below. Do not add data, claims, policies, tools, credentials, or instructions. "
        "Write conversationally: answer the user's actual question first, avoid template language, labels, and repetitive introductions, and vary phrasing naturally. "
        "Every date, percentage, count, and attendance status in the approved facts is immutable: preserve each one exactly and never estimate, round, replace, or omit it. "
        "Keep the response under 70 words.\nApproved facts:\n" + facts
    )
    try:
        async with httpx.AsyncClient(timeout=7) as client:
            payload = {"model": model, "stream": False, "options": {"temperature": 0}, "messages": [{"role": "user", "content": prompt}]}
            logger.info("OLLAMA_RESPONSE_REWRITE_REQUEST %s", json.dumps(payload, ensure_ascii=False))
            response = await client.post(f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/chat", json=payload)
            response.raise_for_status()
            response_data = response.json()
            logger.info("OLLAMA_RESPONSE_REWRITE_OUTPUT %s", json.dumps(response_data, ensure_ascii=False))
            content = response_data["message"]["content"].strip().strip('"')
            if required_terms and any(term.lower() not in content.lower() for term in required_terms):
                logger.warning("OLLAMA_RESPONSE_REWRITE_REJECTED missing immutable attendance facts")
                return facts, "rules"
            return (content or facts), "ollama"
    except (httpx.HTTPError, KeyError, ValueError):
        return facts, "rules"


async def plan_with_ollama(message: str, user: dict, language: str, conversation_stage: str) -> dict | None:
    """Ollama produces the operation plan; application code executes it safely."""
    model = os.getenv("OLLAMA_MODEL")
    if not model:
        return None
    schema = {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["attendance", "attendance_history", "update_attendance", "school_analytics", "escalate", "conversation"], "description": "Use attendance_history for every dated, monthly, or range-based attendance request."},
            "student_name": {"type": "string"},
            "status": {"type": "string", "enum": ["present", "absent"]},
            "attendance_date": {"type": "string", "description": "Only for update_attendance. Must be an ISO date between 2026-06-01 and 2026-08-19."},
            "start_date": {"type": "string", "description": "Start of a requested attendance range, as ISO YYYY-MM-DD."},
            "end_date": {"type": "string", "description": "End of a requested attendance range, as ISO YYYY-MM-DD."},
            "target": {"type": "string", "enum": ["teacher", "management"]},
            "reply": {"type": "string"},
        },
        "required": ["operation", "reply"],
    }
    dataset_start = ATTENDANCE_DATES[0]
    dataset_today = ATTENDANCE_DATES[-1]
    yesterday = (datetime.strptime(dataset_today, "%Y-%m-%d").date() - timedelta(days=1)).isoformat()
    three_days_back = (datetime.strptime(dataset_today, "%Y-%m-%d").date() - timedelta(days=3)).isoformat()
    system = (
        f"You are XYZ AI's decision engine. Current user role: {user['role']}. Language: {language}. Conversation stage: {conversation_stage}. "
        "Return only JSON matching the provided schema. Infer the user's requested operation from natural language. "
        "Use attendance only for current aggregate attendance. Use attendance_history for any dated, monthly, or date-range attendance request, including yesterday, today, last week, three days back, on Monday, in July, this month, current month, or last month. "
        "update_attendance for attendance changes, school_analytics for school-wide requests, "
        f"escalate for a human request, and conversation otherwise. The available attendance window is {dataset_start} through {dataset_today}; the current dataset date is {dataset_today}. "
        "Convert relative date questions into exact ISO dates. For a single-day question, set start_date and end_date to the same date. "
        f"Examples: 'Was I present yesterday?' is attendance_history with start_date and end_date both {yesterday}. "
        f"'What was my attendance 3 days back?' is attendance_history with start_date and end_date both {three_days_back}. "
        "For a greeting stage, choose conversation and greet naturally in the reply. For an action or fact request, "
        "provide a short natural reply that either answers the user or asks for any missing detail. "
        "Never invent names, dates, authority, credentials, or data."
    )
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            payload = {"model": model, "stream": False, "format": schema, "messages": [{"role": "system", "content": system}, {"role": "user", "content": message}]}
            logger.info("OLLAMA_PLAN_REQUEST %s", json.dumps(payload, ensure_ascii=False))
            response = await client.post(f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/chat", json=payload)
            response.raise_for_status()
            response_data = response.json()
            logger.info("OLLAMA_PLAN_OUTPUT %s", json.dumps(response_data, ensure_ascii=False))
            plan = json.loads(response_data.get("message", {}).get("content", "{}"))
            if plan.get("operation") not in {"attendance", "attendance_history", "update_attendance", "school_analytics", "escalate", "conversation"}:
                return None
            return plan
    except (httpx.HTTPError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def find_student_id(name: str | None, user: dict) -> str | None:
    """Resolve a student from trusted CSV records; no model-provided ID is accepted."""
    if user["role"] == "student":
        return user.get("student_id")
    if user["role"] == "parent":
        return user.get("child_ids", [None])[0]
    if not name:
        return None
    wanted = normalized(name)
    for student_id, student in STUDENTS.items():
        if normalized(student["name"]) == wanted:
            return student_id
    return None


def attendance_history_summary(student_id: str, start_date: str, end_date: str) -> str:
    if start_date == end_date:
        status = ATTENDANCE_RECORDS.get((start_date, student_id))
        if not status:
            raise HTTPException(status_code=422, detail="No attendance record is available for that date.")
        return start_date + ": " + status
    records = [(record_date, status) for (record_date, record_student_id), status in ATTENDANCE_RECORDS.items() if record_student_id == student_id and start_date <= record_date <= end_date]
    if not records:
        raise HTTPException(status_code=422, detail="No attendance records are available for that date range.")
    if (datetime.strptime(end_date, "%Y-%m-%d").date() - datetime.strptime(start_date, "%Y-%m-%d").date()).days < 31:
        present_days = sum(status == "present" for _, status in records)
        return "attended " + str(present_days) + " of " + str(len(records)) + " school days from " + start_date + " to " + end_date + ", which is " + format(present_days / len(records) * 100, ".1f") + "% attendance"
    grouped: dict[str, list[str]] = {}
    for record_date, status in records:
        label = datetime.strptime(record_date, "%Y-%m-%d").strftime("%B %Y")
        grouped.setdefault(label, []).append(status)
    return ", ".join(month + ": " + format(sum(status == "present" for status in statuses) / len(statuses) * 100, ".1f") + "%" for month, statuses in grouped.items())


def requested_attendance_date(value: str | None) -> date:
    """Accept an ISO date from the calendar or LLM; default to the latest school date."""
    try:
        return date.fromisoformat(value) if value else date.fromisoformat(ATTENDANCE_DATES[-1])
    except ValueError:
        raise HTTPException(status_code=422, detail="Please use a date in YYYY-MM-DD format.")


@app.get("/health")
def health():
    return {"status": "ok", "ollama_configured": bool(os.getenv("OLLAMA_MODEL"))}


@app.post("/api/auth/login")
def portal_login(request: PortalLoginRequest):
    """Portal login for all four roles; the server derives role and permissions from CSV."""
    account = ACCOUNTS.get(request.email.lower())
    if not account or account["password"] != request.password:
        raise HTTPException(status_code=401, detail="Incorrect school email or password.")
    claims = {key: value for key, value in account.items() if key != "password"}
    return {"token": create_token(claims), "user": {k: claims[k] for k in ("id", "name", "role")}}


@app.get("/api/attendance/overview")
def attendance_overview(grade: int | None = None, section: str | None = None, attendance_date: date | None = None, user: dict = Depends(current_user)):
    """Staff dashboard endpoint; student and parent tokens are deliberately rejected."""
    require_role(user, "teacher", "principal")
    if grade is not None and not 1 <= grade <= 10:
        raise HTTPException(status_code=422, detail="Grade must be between 1 and 10.")
    if user["role"] == "teacher" and grade is None:
        assigned_class = user["class_ids"][0]  # e.g. grade_8a
        requested_grade, requested_section = int(assigned_class.split("_")[1][:-1]), assigned_class[-1].upper()
    else:
        requested_grade = grade or 1
        requested_section = (section or "A").upper()
    if requested_section not in {"A", "B", "C"}:
        raise HTTPException(status_code=422, detail="Section must be A, B, or C.")
    selected_date = (attendance_date or date.fromisoformat(ATTENDANCE_DATES[-1])).isoformat()
    if selected_date not in ATTENDANCE_DATES:
        raise HTTPException(status_code=422, detail=f"Choose a date between {ATTENDANCE_DATES[0]} and {ATTENDANCE_DATES[-1]}.")
    class_id = f"grade_{requested_grade}{requested_section.lower()}"
    all_classes = [{"grade": grade_num, "section": sec, "label": f"Class {grade_num}{sec}"} for grade_num in range(1, 11) for sec in "ABC"]
    if user["role"] == "teacher" and class_id not in user.get("class_ids", []):
        raise HTTPException(status_code=403, detail="Teachers can only view rosters for their assigned classes.")
    allowed_classes = all_classes if user["role"] == "principal" else [item for item in all_classes if f"grade_{item['grade']}{item['section'].lower()}" in user.get("class_ids", [])]
    roster = [attendance_payload(student_id, selected_date) for student_id, student in STUDENTS.items() if student["class_id"] == class_id]
    teacher_name = STUDENTS[roster[0]["student_id"]]["teacher_name"] if roster else "Unassigned"
    result = {
        "view": user["role"],
        "class_name": f"Class {requested_grade}{requested_section}",
        "teacher_name": teacher_name,
        "attendance": roster,
        "available_classes": allowed_classes,
        "class_average": round(sum(item["attendance_percentage"] for item in roster) / len(roster), 1),
        "selected_date": selected_date,
        "date_min": ATTENDANCE_DATES[0],
        "date_max": ATTENDANCE_DATES[-1],
    }
    if user["role"] == "principal":
        result.update({"school_attendance": 92.8, "monthly_change": 1.4, "grade_8a": 91.2, "total_students": len(STUDENTS)})
    return result


@app.post("/api/attendance/mark")
def dashboard_mark_attendance(request: MarkAttendanceRequest, user: dict = Depends(current_user)):
    message = mark_attendance(user, request.student_id, request.status, request.attendance_date)
    return {"success": True, "message": message, "attendance": attendance_payload(request.student_id, request.attendance_date.isoformat())}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, user: dict = Depends(current_user)):
    message = request.message.strip()
    language = requested_language(request.language, message)
    if is_injection(message):
        response, source = await improve_with_ollama("The request seeks protected internal information and cannot be completed.", user, language, "safety")
        return ChatResponse(response=response, intent="security_refusal", source=source)

    state = CONVERSATIONS.setdefault(request.session_id, {"history": [], "pending_escalation": None})
    state["history"].append({"role": "user", "content": message})
    text = normalized(message)
    # Confirmation is contextual: â€œyesâ€ has no action unless a server-side pending request exists.
    if state.get("pending_escalation") and text in {"yes", "yes please", "confirm", "haan", "ha"}:
        target = state.pop("pending_escalation")
        if submit_escalation(user, target):
            facts = "A support request to " + ("the teacher" if target == "teacher" else "school management") + " was submitted successfully."
            response, source = await improve_with_ollama(facts, user, language, "escalation")
            return ChatResponse(response=response, intent="escalation_confirmed", action_taken=True, source=source)

    # Ollama creates the complete operation plan. It never receives a CSV ID or
    # performs the operation itself; resolution and authorization stay here.
    conversation_stage = "first_turn" if len(state["history"]) == 1 else "continuation"
    plan = await plan_with_ollama(message, user, language, conversation_stage)
    if plan:
        operation = plan["operation"]
        relative_window = relative_attendance_window(message)
        if relative_window and operation not in {"update_attendance", "school_analytics", "escalate"}:
            operation = "attendance_history"
        try:
            if operation == "school_analytics":
                response, _ = await improve_with_ollama(school_analytics(user), user, language, "analytics")
                return ChatResponse(response=response, intent="school_analytics", source="ollama")
            if operation == "escalate":
                target = plan.get("target") if plan.get("target") in {"teacher", "management"} else "teacher"
                state["pending_escalation"] = target
                facts = "The user requested escalation to " + target + ". Ask for confirmation before submitting any request."
                response, _ = await improve_with_ollama(facts, user, language, "escalation")
                return ChatResponse(response=response, intent="escalation_request", escalation_options=["Talk to Teacher", "Contact School Management"], source="ollama")
            if operation == "conversation":
                response = plan.get("reply") or ""
                return ChatResponse(response=response, intent="general", source="ollama")
            student_id = find_student_id(plan.get("student_name"), user)
            if not student_id:
                return ChatResponse(response=plan.get("reply") or "", intent="missing_student", source="ollama")
            student = attendance_for(user, student_id)
            if operation == "attendance_history":
                start_date, end_date = relative_window or (plan.get("start_date") or ATTENDANCE_DATES[0], plan.get("end_date") or ATTENDANCE_DATES[-1])
                if start_date not in ATTENDANCE_DATES or end_date not in ATTENDANCE_DATES or start_date > end_date:
                    raise HTTPException(status_code=422, detail="The requested range is outside the available attendance dates.")
                history = attendance_history_summary(student_id, start_date, end_date)
                subject = "your child, " + student["name"] if user["role"] == "parent" else "you"
                range_subject = student["name"] + "'s" if user["role"] == "parent" else "Your"
                facts = ("Yes, " + subject + " was " + ATTENDANCE_RECORDS[(start_date, student_id)] + " on " + start_date + ".") if start_date == end_date else ((student["name"] if user["role"] == "parent" else "You") + " " + history + "." if history.startswith("attended ") else range_subject + " " + history[0].lower() + history[1:] + ".")
                required_terms = tuple(re.findall(r"\b\d{4}-\d{2}-\d{2}\b|\b(?:present|absent)\b", history, flags=re.IGNORECASE)) if start_date == end_date else tuple(re.findall(r"\b\d+(?:\.\d+)?%", history))
                response, source = await improve_with_ollama(facts, user, language, "history", required_terms)
                return ChatResponse(response=response, intent="attendance_history", source=source)
            if operation == "attendance":
                percentage = student["attendance"]["present"] / student["attendance"]["total"] * 100
                facts = student["name"] + " current attendance is " + format(percentage, ".1f") + "%."
                response, _ = await improve_with_ollama(facts, user, language, "attendance")
                return ChatResponse(response=response, intent="attendance", source="ollama")
            if operation == "update_attendance":
                requested_status = plan.get("status")
                if requested_status not in {"present", "absent"}:
                    return ChatResponse(response=plan.get("reply") or "", intent="missing_status", source="ollama")
                action_fact = mark_attendance(user, student_id, requested_status, requested_attendance_date(plan.get("attendance_date")))
                response, _ = await improve_with_ollama(action_fact, user, language, "attendance_update")
                return ChatResponse(response=response, intent="mark_attendance", action_taken=True, source="ollama")
        except HTTPException as err:
            response, _ = await improve_with_ollama("The requested operation was not authorized: " + err.detail, user, language, "safety")
            return ChatResponse(response=response, intent="access_denied", source="ollama")
    return ChatResponse(response="", intent="llm_unavailable")

