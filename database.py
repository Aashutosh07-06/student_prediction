import mysql.connector


# ==================================================
# DATABASE CONNECTION
# ==================================================

def get_connection():

    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="sr123",
        database="edupredict"
    )


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