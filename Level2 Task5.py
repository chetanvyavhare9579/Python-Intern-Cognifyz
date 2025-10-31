import random
# ---- Step 1: Generate a random text file ----
def create_random_file(filename, num_words):
    words = ["Python", "hello", "world", "code", "program", "data", "file", "test", "AI", "developer"]
    random_text = " ".join(random.choice(words) for _ in range(num_words))

    with open(filename, 'w') as file:
        file.write(random_text)

    print(f"Random file '{filename}' created with {num_words} words!\n")

# ---- Step 2: Count word occurrences in the file ----
def count_words_in_file(filename):
    try:
        with open(filename, 'r') as file:
            text = file.read().lower()  # Convert text to lowercase
            words = text.split()  # Split into words

        word_count = {}
        for word in words:
            word = word.strip(".,!?;:()[]{}\"'")  # Remove punctuation
            if word:
                word_count[word] = word_count.get(word, 0) + 1

        # Sort alphabetically
        sorted_words = dict(sorted(word_count.items()))

        print("Word Count in Alphabetical Order:")
        for word, count in sorted_words.items():
            print(f"{word}: {count}")

    except FileNotFoundError:
        print(f"Error: The file '{filename}' does not exist.")

# ---- Main Program ----
filename = "random_text_file.txt"
num_words = int(input("Enter the number of random words to generate in the file: "))

create_random_file(filename, num_words)
count_words_in_file(filename)
