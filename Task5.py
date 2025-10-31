# Task 5: Level 1

# Function to check if a string is a palindrome
def is_palindrome(text):
    cleaned_text = text.replace(" ", "").lower()
    return cleaned_text == cleaned_text[::-1]

# Take input from the user
string = input("Enter a word or phrase: ")

# Check and display the result
if is_palindrome(string):
    print(f" '{string}' is a palindrome!")
else:
    print(f" '{string}' is not a palindrome.")
