import random
print("Welcome to the Guessing game****")
n=random.randint(1,100)
a=-1
guesses=0

while (a != n):
    guesses+=1
    a=int(input("Guess a number:"))
    if a>n:
        print("Enter a lowest number")
    else:
        print("Enter a highest number")
print(f"You have  guess the number correctly{guesses} attempt.")