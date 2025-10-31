import re

def check_password_strength(password):
    strength = 0
    remarks = ""
    # Check length
    if len(password) < 6:
        remarks = "Password is too short. Minimum 6 characters required."
    elif len(password) >= 8:
        strength += 1

    # Check for uppercase letters
    if re.search(r"[A-Z]", password):
        strength += 1
    else:
        remarks += "\nAdd at least one uppercase letter."

    # Check for lowercase letters
    if re.search(r"[a-z]", password):
        strength += 1
    else:
        remarks += "\nAdd at least one lowercase letter."

    # Check for digits
    if re.search(r"[0-9]", password):
        strength += 1
    else:
        remarks += "\nAdd at least one number."

    # Check for special characters
    if re.search(r"[@$!%*?&]", password):
        strength += 1
    else:
        remarks += "\nAdd at least one special character (@, $, !, %, *, ?, &)."

    # Evaluate strength level
    if strength <= 2:
        level = "Weak"
    elif strength == 3 or strength == 4:
        level = "Moderate"
    else:
        level = "Strong"

    print("\nPassword Strength:", level)
    print(remarks if remarks else "Your password looks strong and secure!")

# ---- Main program ----
password = input("Enter your password: ")
check_password_strength(password)
