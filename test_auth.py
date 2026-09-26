from auth import login_user


email = input("Enter email: ")
password = input("Enter password: ")

user = login_user(
    email,
    password
)

if user:

    print("\nLogin successful!")

    print("User ID:", user["user_id"])
    print("Name:", user["name"])
    print("Email:", user["email"])
    print("Role:", user["role"])

else:

    print("\nInvalid email or password.")