import os
import sqlite3
import bcrypt
from datetime import datetime, timedelta

# Try importing mysql.connector
try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False


# ==================================================
# SQLITE COMPATIBILITY WRAPPERS (FOR CLOUD DEPLOYMENT)
# ==================================================

class SQLiteCursorWrapper:
    """Wraps sqlite3.Cursor to behave like a mysql.connector cursor."""
    def __init__(self, raw_cursor, dictionary=False):
        self.cursor = raw_cursor
        self.dictionary = dictionary

    @property
    def description(self):
        return self.cursor.description

    @property
    def lastrowid(self):
        return self.cursor.lastrowid

    @property
    def rowcount(self):
        return self.cursor.rowcount

    def execute(self, query, params=None):
        # Convert MySQL %s placeholder to SQLite ? placeholder
        converted = query.replace("%s", "?")
        if params is None:
            self.cursor.execute(converted)
        else:
            self.cursor.execute(converted, tuple(params))
        return self

    def executemany(self, query, seq_of_params):
        converted = query.replace("%s", "?")
        self.cursor.executemany(converted, seq_of_params)
        return self

    def fetchone(self):
        row = self.cursor.fetchone()
        if row is None:
            return None
        if self.dictionary:
            return dict(row)
        return tuple(row)

    def fetchall(self):
        rows = self.cursor.fetchall()
        if self.dictionary:
            return [dict(r) for r in rows]
        return [tuple(r) for r in rows]

    def close(self):
        try:
            self.cursor.close()
        except Exception:
            pass


class SQLiteConnectionWrapper:
    """Wraps sqlite3.Connection to provide a MySQL-compatible interface."""
    def __init__(self, db_path="edupredict.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def cursor(self, dictionary=False, **kwargs):
        return SQLiteCursorWrapper(self.conn.cursor(), dictionary=dictionary)

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass


def init_sqlite_db(db_path="edupredict.db"):
    """Initializes tables and seeds baseline demo data if SQLite database is newly created."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()

    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        role TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS student_profiles (
        student_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER UNIQUE,
        roll_number TEXT UNIQUE,
        course TEXT,
        semester INTEGER,
        section TEXT
    );

    CREATE TABLE IF NOT EXISTS teacher_profiles (
        teacher_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER UNIQUE,
        employee_id TEXT UNIQUE,
        department TEXT
    );

    CREATE TABLE IF NOT EXISTS predictions (
        prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        study_hours REAL,
        attendance REAL,
        resources REAL,
        extracurricular INTEGER,
        motivation REAL,
        internet INTEGER,
        gender INTEGER,
        age INTEGER,
        learning_style INTEGER,
        online_courses INTEGER,
        discussions INTEGER,
        assignment_completion REAL,
        exam_score REAL,
        edutech INTEGER,
        stress_level INTEGER,
        predicted_grade REAL,
        prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.commit()

    # Check if users table is populated
    cursor.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]

    if count == 0:
        def hash_pw(pw):
            return bcrypt.hashpw(pw.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

        # 1. Officials
        officials = [
            ("Dr. R. K. Narayanan (Campus Director)", "official@gmail.com", "official123"),
            ("Dr. S. K. Sharma (Dean of Academics)", "dean.academics@edupredict.ai", "dean123"),
            ("Dr. Sunita Rao (HOD - Computer Science)", "hod.cse@edupredict.ai", "hod123"),
            ("Prof. Arvind Patel (Controller of Examinations)", "exam.controller@edupredict.ai", "exam123"),
        ]
        for name, email, pw in officials:
            cursor.execute(
                "INSERT INTO users (name, email, password, role) VALUES (?, ?, ?, 'official')",
                (name, email, hash_pw(pw))
            )

        # 2. Teachers
        teachers = [
            ("Prof. Ananya Sen", "teacher@gmail.com", "teacher123", "EMP-AI-101", "Artificial Intelligence & Data Science"),
            ("Dr. Vikram Verma", "prof.verma@edupredict.ai", "teacher123", "EMP-CS-102", "Computer Science & Engineering"),
            ("Dr. Meera Iyer", "prof.iyer@edupredict.ai", "teacher123", "EMP-IT-103", "Information Technology"),
        ]
        for name, email, pw, emp_id, dept in teachers:
            cursor.execute(
                "INSERT INTO users (name, email, password, role) VALUES (?, ?, ?, 'teacher')",
                (name, email, hash_pw(pw))
            )
            uid = cursor.lastrowid
            cursor.execute(
                "INSERT INTO teacher_profiles (user_id, employee_id, department) VALUES (?, ?, ?)",
                (uid, emp_id, dept)
            )

        # 3. Students
        students_data = [
            ("Student One (Demo)", "student@gmail.com", "student123", "BETN1AI25074", "B.Tech AIML", 3, "D"),
            ("Aarav Sharma", "aarav.sharma@edupredict.ai", "student123", "BETN1AI25001", "B.Tech AIML", 3, "A"),
            ("Diya Patel", "diya.patel@edupredict.ai", "student123", "BETN1AI25002", "B.Tech AIML", 3, "A"),
            ("Ishaan Verma", "ishaan.verma@edupredict.ai", "student123", "BETN1AI25003", "B.Tech AIML", 3, "B"),
            ("Ananya Nair", "ananya.nair@edupredict.ai", "student123", "BETN1AI25004", "B.Tech AIML", 3, "B"),
            ("Kabir Mehta", "kabir.mehta@edupredict.ai", "student123", "BETN1CS25010", "B.Tech CSE", 5, "A"),
            ("Rhea Gupta", "rhea.gupta@edupredict.ai", "student123", "BETN1CS25011", "B.Tech CSE", 5, "A"),
            ("Aditya Rao", "aditya.rao@edupredict.ai", "student123", "BETN1CS25012", "B.Tech CSE", 5, "B"),
            ("Sanya Kapoor", "sanya.kapoor@edupredict.ai", "student123", "BETN1CS25013", "B.Tech CSE", 5, "B"),
            ("Vivaan Joshi", "vivaan.joshi@edupredict.ai", "student123", "BDSC1DS25020", "B.Sc Data Science", 3, "A"),
            ("Tanvi Deshmukh", "tanvi.deshmukh@edupredict.ai", "student123", "BDSC1DS25021", "B.Sc Data Science", 3, "A"),
            ("Karan Malhotra", "karan.malhotra@edupredict.ai", "student123", "BBCA1BC25030", "BCA", 1, "C"),
            ("Meera Sen", "meera.sen@edupredict.ai", "student123", "BBCA1BC25031", "BCA", 1, "C"),
        ]

        student_ids = []
        for name, email, pw, roll, course, sem, sec in students_data:
            cursor.execute(
                "INSERT INTO users (name, email, password, role) VALUES (?, ?, ?, 'student')",
                (name, email, hash_pw(pw))
            )
            uid = cursor.lastrowid
            cursor.execute(
                "INSERT INTO student_profiles (user_id, roll_number, course, semester, section) VALUES (?, ?, ?, ?, ?)",
                (uid, roll, course, sem, sec)
            )
            student_ids.append(cursor.lastrowid)

        # Baseline chronological predictions for demo student
        if student_ids:
            demo_sid = student_ids[0]
            base_date = datetime.now() - timedelta(days=60)
            sample_sessions = [
                (7.0, 62.0, 52.0, 58.0, 0, 0, 2, 0, 62.4),
                (10.5, 71.0, 63.0, 68.0, 1, 1, 2, 14, 71.2),
                (14.0, 79.0, 72.0, 78.0, 1, 1, 1, 28, 78.5),
                (18.0, 87.0, 83.0, 88.0, 2, 2, 1, 42, 85.1),
                (22.0, 94.0, 92.0, 96.0, 2, 2, 0, 56, 92.8),
            ]
            for hours, att, exam, assign, res, motiv, stress, offset, grade in sample_sessions:
                dt_str = (base_date + timedelta(days=offset, hours=10)).strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute("""
                    INSERT INTO predictions (
                        student_id, study_hours, attendance, resources, extracurricular,
                        motivation, internet, gender, age, learning_style, online_courses,
                        discussions, assignment_completion, exam_score, edutech, stress_level,
                        predicted_grade, prediction_date
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    demo_sid, hours, att, res, 1,
                    motiv, 1, 1, 20, 1, 2,
                    1, assign, exam, 1, stress,
                    grade, dt_str
                ))

        conn.commit()

    conn.close()


# ==================================================
# DATABASE CONNECTION FACTORY
# ==================================================

def get_connection():
    """
    Connects to MySQL if available (checking Streamlit secrets, environment variables,
    or local defaults). If MySQL is unavailable (e.g. running in Cloud without remote MySQL),
    automatically and gracefully falls back to an embedded SQLite database.
    """
    # 1. Resolve configuration from Streamlit secrets, Env vars, or Defaults
    db_host = os.getenv("DB_HOST")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_name = os.getenv("DB_DATABASE", os.getenv("DB_NAME", "edupredict"))
    db_port = int(os.getenv("DB_PORT", 3306))

    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if "mysql" in st.secrets:
                sec = st.secrets["mysql"]
                db_host = sec.get("host", db_host)
                db_user = sec.get("user", db_user)
                db_password = sec.get("password", db_password)
                db_name = sec.get("database", db_name)
                db_port = int(sec.get("port", db_port))
            elif "db" in st.secrets:
                sec = st.secrets["db"]
                db_host = sec.get("host", db_host)
                db_user = sec.get("user", db_user)
                db_password = sec.get("password", db_password)
                db_name = sec.get("database", db_name)
                db_port = int(sec.get("port", db_port))
    except Exception:
        pass

    db_host = db_host or "localhost"
    db_user = db_user or "root"
    db_password = db_password if db_password is not None else "sr123"

    # 2. Attempt MySQL connection
    if MYSQL_AVAILABLE:
        try:
            conn = mysql.connector.connect(
                host=db_host,
                user=db_user,
                password=db_password,
                database=db_name,
                port=db_port,
                connection_timeout=3
            )
            return conn
        except Exception:
            # MySQL connection failed (e.g. localhost on cloud, or bad credentials)
            pass

    # 3. Graceful fallback to SQLite (guarantees zero-crash cloud deployment)
    sqlite_path = os.path.join(os.path.dirname(__file__), "edupredict.db")
    init_sqlite_db(sqlite_path)
    return SQLiteConnectionWrapper(sqlite_path)


# ==================================================
# STUDENT PROFILE
# ==================================================

def get_student_profile(user_id):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    query = """
    SELECT
        sp.student_id,
        sp.roll_number,
        u.name,
        u.email,
        sp.course,
        sp.semester,
        sp.section
    FROM student_profiles sp
    JOIN users u
        ON sp.user_id = u.user_id
    WHERE sp.user_id = %s
    """

    cursor.execute(query, (user_id,))
    result = cursor.fetchone()

    cursor.close()
    conn.close()

    return result


# ==================================================
# TEACHER PROFILE
# ==================================================

def get_teacher_profile(user_id):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    query = """
    SELECT
        tp.teacher_id,
        tp.employee_id,
        tp.department,
        u.name,
        u.email
    FROM teacher_profiles tp
    JOIN users u
        ON tp.user_id = u.user_id
    WHERE tp.user_id = %s
    """

    cursor.execute(query, (user_id,))
    result = cursor.fetchone()

    cursor.close()
    conn.close()

    return result


# ==================================================
# SAVE PREDICTION
# ==================================================

def save_prediction(
    student_id,
    study_hours,
    attendance,
    resources,
    extracurricular,
    motivation,
    internet,
    gender,
    age,
    learning_style,
    online_courses,
    discussions,
    assignment_completion,
    exam_score,
    edutech,
    stress_level,
    predicted_grade
):
    conn = get_connection()
    cursor = conn.cursor()

    query = """
    INSERT INTO predictions
    (
        student_id,
        study_hours,
        attendance,
        resources,
        extracurricular,
        motivation,
        internet,
        gender,
        age,
        learning_style,
        online_courses,
        discussions,
        assignment_completion,
        exam_score,
        edutech,
        stress_level,
        predicted_grade
    )
    VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s, %s
    )
    """

    cursor.execute(
        query,
        (
            student_id,
            study_hours,
            attendance,
            resources,
            extracurricular,
            motivation,
            internet,
            gender,
            age,
            learning_style,
            online_courses,
            discussions,
            assignment_completion,
            exam_score,
            edutech,
            stress_level,
            predicted_grade
        )
    )

    conn.commit()
    cursor.close()
    conn.close()


# ==================================================
# STUDENT'S OWN PREDICTIONS
# ==================================================

def get_student_predictions(student_id):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    query = """
    SELECT
        prediction_id,
        study_hours,
        attendance,
        resources,
        extracurricular,
        motivation,
        internet,
        gender,
        age,
        learning_style,
        online_courses,
        discussions,
        assignment_completion,
        exam_score,
        edutech,
        stress_level,
        predicted_grade,
        prediction_date
    FROM predictions
    WHERE student_id = %s
    ORDER BY prediction_date DESC
    """

    cursor.execute(query, (student_id,))
    result = cursor.fetchall()

    cursor.close()
    conn.close()

    return result


# ==================================================
# ALL STUDENTS
# ==================================================

def get_all_students():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    query = """
    SELECT
        sp.student_id,
        sp.roll_number,
        u.name,
        u.email,
        sp.course,
        sp.semester,
        sp.section
    FROM student_profiles sp
    JOIN users u
        ON sp.user_id = u.user_id
    ORDER BY sp.roll_number
    """

    cursor.execute(query)
    result = cursor.fetchall()

    cursor.close()
    conn.close()

    return result


# ==================================================
# ALL PREDICTIONS
# ==================================================

def get_all_predictions():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    query = """
    SELECT
        p.prediction_id,
        sp.roll_number,
        u.name,
        p.study_hours,
        p.attendance,
        p.resources,
        p.extracurricular,
        p.motivation,
        p.internet,
        p.gender,
        p.age,
        p.learning_style,
        p.online_courses,
        p.discussions,
        p.assignment_completion,
        p.exam_score,
        p.edutech,
        p.stress_level,
        p.predicted_grade,
        p.prediction_date
    FROM predictions p
    JOIN student_profiles sp
        ON p.student_id = sp.student_id
    JOIN users u
        ON sp.user_id = u.user_id
    ORDER BY p.prediction_date DESC
    """

    cursor.execute(query)
    result = cursor.fetchall()

    cursor.close()
    conn.close()

    return result


# ==================================================
# ALL USERS
# ==================================================

def get_all_users():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    query = """
    SELECT
        user_id,
        name,
        email,
        role,
        created_at
    FROM users
    ORDER BY created_at DESC
    """

    cursor.execute(query)
    result = cursor.fetchall()

    cursor.close()
    conn.close()

    return result