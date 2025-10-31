# Task: Number Guesser Game
import random
# Generate a random number between 1 and 100
number_to_guess = random.randint(1, 100)
print("🎯 Welcome to the Number Guessing Game!")
print("I'm thinking of a number between 1 and 100... Can you guess it?")
# Loop until the user guesses correctly
while True:
    try:
        guess = int(input("Enter your guess: "))

        if guess < number_to_guess:
            print("📉Too low! Try again.")
        elif guess > number_to_guess:
            print("📈Too high! Try again.")
        else:
            print(f"Congratulations! You guessed the number {number_to_guess} correctly!")
            break
    except ValueError:
        print("🎉 Please enter a valid number.")
