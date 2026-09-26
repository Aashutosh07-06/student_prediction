import bcrypt
import random
from datetime import datetime, timedelta
import joblib
import pandas as pd
import numpy as np

from database import get_connection

FEATURES = [
    "StudyHours", "Attendance", "Resources", "Extracurricular", "Motivation",
    "Internet", "Gender", "Age", "LearningStyle", "OnlineCourses",
    "Discussions", "AssignmentCompletion", "ExamScore", "EduTech", "StressLevel"
]

def hash_pw(pw):
    return bcrypt.hashpw(pw.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def seed():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    loaded_model = joblib.load("model/student_model.pkl")

    print("--- Seeding Database ---")

    # 1. Officials
    officials = [
        ("Dr. R. K. Narayanan (Campus Director)", "official@gmail.com", "official123"),
        ("Dr. S. K. Sharma (Dean of Academics)", "dean.academics@edupredict.ai", "dean123"),
        ("Dr. Sunita Rao (HOD - Computer Science)", "hod.cse@edupredict.ai", "hod123"),
        ("Prof. Arvind Patel (Controller of Examinations)", "exam.controller@edupredict.ai", "exam123"),
    ]

    for name, email, pw in officials:
        cursor.execute("SELECT user_id FROM users WHERE email = %s", (email,))
        if not cursor.fetchone():
            cursor.execute(
                "INSERT INTO users (name, email, password, role) VALUES (%s, %s, %s, 'official')",
                (name, email, hash_pw(pw))
            )
            print(f"Created official: {name}")

    # 2. Teachers
    teachers = [
        ("Prof. Ananya Sen", "teacher@gmail.com", "teacher123", "EMP-AI-101", "Artificial Intelligence & Data Science"),
        ("Dr. Vikram Verma", "prof.verma@edupredict.ai", "teacher123", "EMP-CS-102", "Computer Science & Engineering"),
        ("Dr. Meera Iyer", "prof.iyer@edupredict.ai", "teacher123", "EMP-IT-103", "Information Technology"),
    ]

    for name, email, pw, emp_id, dept in teachers:
        cursor.execute("SELECT user_id FROM users WHERE email = %s", (email,))
        row = cursor.fetchone()
        if not row:
            cursor.execute(
                "INSERT INTO users (name, email, password, role) VALUES (%s, %s, %s, 'teacher')",
                (name, email, hash_pw(pw))
            )
            uid = cursor.lastrowid
            cursor.execute(
                "INSERT INTO teacher_profiles (user_id, employee_id, department) VALUES (%s, %s, %s)",
                (uid, emp_id, dept)
            )
            print(f"Created teacher: {name}")
        else:
            uid = row["user_id"]
            cursor.execute("SELECT teacher_id FROM teacher_profiles WHERE user_id = %s", (uid,))
            if not cursor.fetchone():
                cursor.execute(
                    "INSERT INTO teacher_profiles (user_id, employee_id, department) VALUES (%s, %s, %s)",
                    (uid, emp_id, dept)
                )

    # 3. Students
    students_data = [
        # (Name, Email, Password, RollNumber, Course, Semester, Section, Trajectory Type)
        ("Student One (Demo)", "student@gmail.com", "student123", "BETN1AI25074", "B.Tech AIML", 3, "D", "improving_fast"),
        ("Aarav Sharma", "aarav.sharma@edupredict.ai", "student123", "BETN1AI25001", "B.Tech AIML", 3, "A", "steady_high"),
        ("Diya Patel", "diya.patel@edupredict.ai", "student123", "BETN1AI25002", "B.Tech AIML", 3, "A", "improving"),
        ("Ishaan Verma", "ishaan.verma@edupredict.ai", "student123", "BETN1AI25003", "B.Tech AIML", 3, "B", "fluctuating"),
        ("Ananya Nair", "ananya.nair@edupredict.ai", "student123", "BETN1AI25004", "B.Tech AIML", 3, "B", "improving"),
        ("Kabir Mehta", "kabir.mehta@edupredict.ai", "student123", "BETN1CS25010", "B.Tech CSE", 5, "A", "steady_moderate"),
        ("Rhea Gupta", "rhea.gupta@edupredict.ai", "student123", "BETN1CS25011", "B.Tech CSE", 5, "A", "improving_fast"),
        ("Aditya Rao", "aditya.rao@edupredict.ai", "student123", "BETN1CS25012", "B.Tech CSE", 5, "B", "at_risk"),
        ("Sanya Kapoor", "sanya.kapoor@edupredict.ai", "student123", "BETN1CS25013", "B.Tech CSE", 5, "B", "struggling"),
        ("Vivaan Joshi", "vivaan.joshi@edupredict.ai", "student123", "BDSC1DS25020", "B.Sc Data Science", 3, "A", "improving"),
        ("Tanvi Deshmukh", "tanvi.deshmukh@edupredict.ai", "student123", "BDSC1DS25021", "B.Sc Data Science", 3, "A", "steady_high"),
        ("Karan Malhotra", "karan.malhotra@edupredict.ai", "student123", "BBCA1BC25030", "BCA", 1, "C", "steady_moderate"),
        ("Meera Sen", "meera.sen@edupredict.ai", "student123", "BBCA1BC25031", "BCA", 1, "C", "improving_fast"),
    ]

    student_records = []

    for name, email, pw, roll, course, sem, sec, traj in students_data:
        cursor.execute("SELECT user_id FROM users WHERE email = %s", (email,))
        row = cursor.fetchone()
        if not row:
            cursor.execute(
                "INSERT INTO users (name, email, password, role) VALUES (%s, %s, %s, 'student')",
                (name, email, hash_pw(pw))
            )
            uid = cursor.lastrowid
            cursor.execute(
                "INSERT INTO student_profiles (user_id, roll_number, course, semester, section) VALUES (%s, %s, %s, %s, %s)",
                (uid, roll, course, sem, sec)
            )
            sid = cursor.lastrowid
            print(f"Created student profile: {name} (Roll: {roll})")
        else:
            uid = row["user_id"]
            cursor.execute("SELECT student_id FROM student_profiles WHERE user_id = %s", (uid,))
            sp_row = cursor.fetchone()
            if not sp_row:
                cursor.execute(
                    "INSERT INTO student_profiles (user_id, roll_number, course, semester, section) VALUES (%s, %s, %s, %s, %s)",
                    (uid, roll, course, sem, sec)
                )
                sid = cursor.lastrowid
            else:
                sid = sp_row["student_id"]

        student_records.append((sid, name, roll, traj))

    conn.commit()

    # Clear old predictions so we have a clean, realistic, chronological history
    cursor.execute("DELETE FROM predictions")
    conn.commit()
    print("Cleaned old prediction records for fresh seed.")

    # 4. Generate multi-session sequential predictions showing improvement
    base_date = datetime.now() - timedelta(days=60)

    for sid, name, roll, traj in student_records:
        if traj == "improving_fast":
            # 5 sequential sessions showing dramatic improvement (e.g. Grade D -> Grade C -> Grade B -> Grade B+ -> Grade A)
            sessions = [
                # study_hours, attendance, exam_score, assignments, resources, motivation, stress
                (7.0, 62.0, 52.0, 58.0, 0, 0, 2, 0),
                (10.5, 71.0, 63.0, 68.0, 1, 1, 2, 14),
                (14.0, 79.0, 72.0, 78.0, 1, 1, 1, 28),
                (18.0, 87.0, 83.0, 88.0, 2, 2, 1, 42),
                (22.0, 94.0, 92.0, 96.0, 2, 2, 0, 56),
            ]
        elif traj == "improving":
            sessions = [
                (11.0, 70.0, 65.0, 70.0, 1, 1, 2, 0),
                (14.0, 78.0, 74.0, 80.0, 1, 1, 1, 20),
                (17.5, 86.0, 84.0, 88.0, 2, 2, 1, 40),
                (20.0, 92.0, 90.0, 94.0, 2, 2, 0, 55),
            ]
        elif traj == "steady_high":
            sessions = [
                (19.0, 91.0, 88.0, 92.0, 2, 2, 1, 0),
                (20.5, 93.0, 90.0, 94.0, 2, 2, 0, 25),
                (22.0, 95.0, 93.0, 97.0, 2, 2, 0, 50),
            ]
        elif traj == "steady_moderate":
            sessions = [
                (12.0, 74.0, 68.0, 72.0, 1, 1, 1, 0),
                (13.5, 76.0, 71.0, 75.0, 1, 1, 1, 28),
                (14.0, 78.0, 73.0, 77.0, 1, 1, 1, 52),
            ]
        elif traj == "fluctuating":
            sessions = [
                (15.0, 80.0, 75.0, 80.0, 1, 1, 1, 0),
                (10.0, 68.0, 62.0, 65.0, 1, 0, 2, 22),
                (16.5, 84.0, 80.0, 85.0, 2, 2, 1, 48),
            ]
        elif traj == "at_risk":
            sessions = [
                (9.0, 64.0, 58.0, 60.0, 0, 0, 2, 0),
                (7.5, 58.0, 52.0, 54.0, 0, 0, 2, 25),
                (6.0, 50.0, 47.0, 48.0, 0, 0, 2, 50),
            ]
        else: # struggling
            sessions = [
                (5.0, 48.0, 45.0, 45.0, 0, 0, 2, 0),
                (6.0, 52.0, 49.0, 50.0, 0, 0, 2, 30),
                (5.5, 49.0, 46.0, 48.0, 0, 0, 2, 55),
            ]

        for s_idx, (hours, att, exam, assign, res, motiv, stress, day_offset) in enumerate(sessions):
            pred_date = base_date + timedelta(days=day_offset, hours=random.randint(9, 17))
            row_dict = {
                "StudyHours": float(hours),
                "Attendance": float(att),
                "Resources": int(res),
                "Extracurricular": 1 if motiv > 0 else 0,
                "Motivation": int(motiv),
                "Internet": 1,
                "Gender": random.choice([0, 1]),
                "Age": random.choice([19, 20, 21]),
                "LearningStyle": random.choice([0, 1, 2]),
                "OnlineCourses": int(motiv + 1),
                "Discussions": 1 if motiv > 0 else 0,
                "AssignmentCompletion": float(assign),
                "ExamScore": float(exam),
                "EduTech": 1 if res > 0 else 0,
                "StressLevel": int(stress)
            }

            # Predict with ML Model
            df_row = pd.DataFrame([row_dict])[FEATURES]
            pred_grade = float(loaded_model.predict(df_row)[0])

            insert_q = """
                INSERT INTO predictions (
                    student_id, study_hours, attendance, resources, extracurricular,
                    motivation, internet, gender, age, learning_style, online_courses,
                    discussions, assignment_completion, exam_score, edutech, stress_level,
                    predicted_grade, prediction_date
                )
                VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s
                )
            """
            params = (
                sid, row_dict["StudyHours"], row_dict["Attendance"], row_dict["Resources"],
                row_dict["Extracurricular"], row_dict["Motivation"], row_dict["Internet"],
                row_dict["Gender"], row_dict["Age"], row_dict["LearningStyle"],
                row_dict["OnlineCourses"], row_dict["Discussions"], row_dict["AssignmentCompletion"],
                row_dict["ExamScore"], row_dict["EduTech"], row_dict["StressLevel"],
                pred_grade, pred_date.strftime("%Y-%m-%d %H:%M:%S")
            )
            cursor.execute(insert_q, params)

    conn.commit()

    cursor.execute("SELECT COUNT(*) as c FROM predictions")
    total_preds = cursor.fetchone()["c"]
    print(f"✅ Successfully seeded {total_preds} chronological predictions across {len(student_records)} students!")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    seed()
