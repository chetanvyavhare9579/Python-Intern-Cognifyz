def generate_fibonacci(n):
    # Check if the number of terms is valid
    if n <= 0:
        print("Please enter a positive integer.")
    elif n == 1:
        print("Fibonacci sequence up to 1 term:")
        print(0)
    else:
        print(f"Fibonacci sequence up to {n} terms:")
        first, second = 0, 1
        print(first, second, end=" ")
        for i in range(2, n):
            next_num = first + second
            print(next_num, end=" ")
            first, second = second, next_num

# ---- Main Program ----
terms = int(input("Enter the number of terms: "))
generate_fibonacci(terms)
