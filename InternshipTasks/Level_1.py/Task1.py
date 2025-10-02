# Level 1: Task 1
# Task: String Reversal

# Function to reverse a string
def string_reverse(s):
    return s[::-1]   # slicing method

# input from user
user_input = input("Enter a string: ")

# function call
string_reverse = string_reverse(user_input)

# output
print("Reversed string:", string_reverse)