"""Generate demo CSV data: 1,200 students and role-based portal accounts."""

import csv
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

FIRST_NAMES = [
    "Aarav", "Aditi", "Arjun", "Ananya", "Vihaan", "Diya", "Kabir", "Isha", "Rohan", "Kavya",
    "Ishaan", "Meera", "Advait", "Saanvi", "Reyansh", "Myra", "Aryan", "Nisha", "Dev", "Tara",
    "Kian", "Riya", "Samar", "Aanya", "Neil", "Pihu", "Vivaan", "Ira", "Yash", "Kiara",
    "Aditya", "Navya", "Krish", "Avni", "Manav", "Siya", "Rudra", "Jiya", "Atharv", "Vanya",
]
LAST_NAMES = ["Sharma", "Verma", "Iyer", "Patel", "Khan", "Reddy", "Singh", "Gupta"]
TEACHER_NAMES = [
    "Asha Nair", "Bharat Joshi", "Charu Menon", "Deepak Rao", "Esha Kapoor", "Farah Ali", "Gaurav Shah", "Heena Das", "Inder Pal", "Jyoti Bose",
    "Karan Malhotra", "Lata Pillai", "Manoj Bhat", "Nandita Roy", "Om Prakash", "Pooja Jain", "Qasim Khan", "Renu Sethi", "Suresh Iyer", "Tanvi Arora",
    "Umesh Gupta", "Vani Reddy", "Wasim Ali", "Xavier Dsouza", "Yamini Shah", "Zoya Khan", "Amit Kulkarni", "Bhavna Mehta", "Chetan Naidu", "Divya Menon",
]

students_path = DATA / "students.csv"
users_path = DATA / "users.csv"
attendance_records_path = DATA / "attendance_records.csv"
parent_rows: list[dict[str, str]] = []
attendance_dates = [date(2026, 6, 1) + timedelta(days=offset) for offset in range((date(2026, 8, 19) - date(2026, 6, 1)).days + 1)]

with students_path.open("w", newline="", encoding="utf-8") as handle, attendance_records_path.open("w", newline="", encoding="utf-8") as records_handle:
    writer = csv.DictWriter(handle, fieldnames=["student_id", "name", "grade", "section", "teacher_id", "teacher_name", "present_days", "total_days", "june_attendance", "july_attendance", "august_attendance"])
    records_writer = csv.DictWriter(records_handle, fieldnames=["date", "student_id", "status"])
    writer.writeheader()
    records_writer.writeheader()
    teacher_index = 0
    for grade in range(1, 11):
        for section in "ABC":
            teacher_id = f"teacher_{grade}{section.lower()}"
            teacher_name = TEACHER_NAMES[teacher_index]
            for roll in range(1, 41):
                student_id = f"g{grade}{section.lower()}_{roll:02d}"
                name = f"{FIRST_NAMES[roll - 1]} {LAST_NAMES[(roll + grade) % len(LAST_NAMES)]}"
                total_days, present_days = 62, 53 + ((grade * 7 + roll * 3 + ord(section)) % 9)
                if grade == 8 and section == "A" and roll == 1:
                    student_id, name, total_days, present_days = "rahul_101", "Rahul Sharma", 57, 52
                monthly = {"june_attendance": 85 + ((roll + grade) % 12), "july_attendance": 84 + ((roll * 2 + grade) % 13), "august_attendance": 86 + ((roll * 3 + grade) % 11)}
                if student_id == "rahul_101":
                    monthly = {"june_attendance": 89.5, "july_attendance": 92.0, "august_attendance": 91.2}
                writer.writerow({"student_id": student_id, "name": name, "grade": grade, "section": section, "teacher_id": teacher_id, "teacher_name": teacher_name, "present_days": present_days, "total_days": total_days, **monthly})
                for attendance_date in attendance_dates:
                    is_absent = (roll * 11 + grade * 7 + ord(section) + attendance_date.day) % 17 == 0
                    records_writer.writerow({"date": attendance_date.isoformat(), "student_id": student_id, "status": "absent" if is_absent else "present"})
                parent_email = "parent.priya@xyzschool.edu" if student_id == "rahul_101" else f"parent.{student_id}@xyzschool.edu"
                parent_name = "Priya Sharma" if student_id == "rahul_101" else f"Parent of {name}"
                parent_rows.append({"email": parent_email, "password": "parent123", "role": "parent", "name": parent_name, "student_id": "", "child_ids": student_id, "class_ids": ""})
            teacher_index += 1

with users_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["email", "password", "role", "name", "student_id", "child_ids", "class_ids"])
    writer.writeheader()
    writer.writerow({"email": "student.rahul@xyzschool.edu", "password": "student123", "role": "student", "name": "Rahul Sharma", "student_id": "rahul_101", "child_ids": "", "class_ids": ""})
    writer.writerows(parent_rows)
    writer.writerow({"email": "principal.mehta@xyzschool.edu", "password": "principal123", "role": "principal", "name": "Dr. Mehta", "student_id": "", "child_ids": "", "class_ids": ""})
    teacher_index = 0
    for grade in range(1, 11):
        for section in "ABC":
            writer.writerow({"email": f"teacher.{grade}{section.lower()}@xyzschool.edu", "password": f"teacher{grade}{section.lower()}123", "role": "teacher", "name": TEACHER_NAMES[teacher_index], "student_id": "", "child_ids": "", "class_ids": f"grade_{grade}{section.lower()}"})
            teacher_index += 1

print(f"Created {students_path} and {users_path}")
