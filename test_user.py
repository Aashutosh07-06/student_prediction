import bcrypt
from database import get_connection


def create_user(name, email, password, role):

    conn = get_connection()
    cursor = conn.cursor()

    hashed_password = bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")

    query = """
    INSERT INTO users (name, email, password, role)
    VALUES (%s, %s, %s, %s)
    """

    cursor.execute(
        query,
        (name, email, hashed_password, role)
    )

    conn.commit()

    user_id = cursor.lastrowid

    cursor.close()
    conn.close()

    return user_id


student_id = create_user(
    "Student One",
    "student@gmail.com",
    "student123",
    "student"
)

teacher_id = create_user(
    "Teacher One",
    "teacher@gmail.com",
    "teacher123",
    "teacher"
)

official_id = create_user(
    "Official One",
    "official@gmail.com",
    "official123",
    "official"
)

print("Users created successfully!")

print("Student User ID:", student_id)
print("Teacher User ID:", teacher_id)
print("Official User ID:", official_id)