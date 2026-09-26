from database import get_connection

try:
    conn = get_connection()

    if conn.is_connected():
        print("MySQL connection successful!")

    conn.close()

except Exception as e:
    print("Database connection failed.")
    print(e)