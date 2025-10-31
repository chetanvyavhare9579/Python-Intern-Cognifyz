# Level 1: Task 3
# Task: Email validator

def validate_email(email):
    if '@' not in email:
        return False

    parts = email.split('@')
    if len(parts) != 2:
        return False

    username = parts[0]
    domain = parts[1]

    if username == "" or domain == "":
        return False

    if '.' not in domain:
        return False

    if domain.startswith('.') or domain.endswith('.'):
        return False

    return True

emails = [
    "user@example.com",
    "hello.world@domain.co.in",
    "user123@gmail",
    "user@.com",
    "@example.com",
    "userexample.com"
]

for e in emails:
    if validate_email(e):
        print(f"{e} → Valid Email")
    else:
        print(f"{e} → Invalid Email")
