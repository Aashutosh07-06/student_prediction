import bcrypt
from database import get_connection


def register_user(name, email, password, role):

    conn = get_connection()
    cursor = conn.cursor()

    hashed_password = bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")

    query = """
    INSERT INTO users
    (name, email, password, role)
    VALUES (%s, %s, %s, %s)
    """

    try:

        cursor.execute(
            query,
            (name, email, hashed_password, role)
        )

        conn.commit()

        user_id = cursor.lastrowid

        return True, user_id

    except Exception as e:

        return False, str(e)

    finally:

        cursor.close()
        conn.close()


def login_user(email, password):

    conn = get_connection()

    cursor = conn.cursor(
        dictionary=True
    )

    query = """
    SELECT *
    FROM users
    WHERE email = %s
    """

    cursor.execute(
        query,
        (email,)
    )

    user = cursor.fetchone()

    cursor.close()
    conn.close()

    if user is None:
        return None

    stored_password = user["password"]

    if bcrypt.checkpw(
        password.encode("utf-8"),
        stored_password.encode("utf-8")
    ):

        return user

    return None