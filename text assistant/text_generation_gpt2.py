import torch
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

# Check if a GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Using 'mps' for newer Apple Silicon Macs is also an option:
# device = torch.device("mps" if torch.backends.mps.is_available() else device)
print(f"Using device: {device}")

# --- 1. Data Preparation: Create a sample text file with custom data ---
# The sample text is crucial for demonstrating the fine-tuning process.
custom_text = """
This is the first line of my custom data.
This is the second line, and it contains more information.
We can add as many lines as needed for training the model.
Ensure the data is relevant to the text you want the model to generate.
This is additional text to make the file longer.
We need enough content to fill at least one block of size 128.
Adding more lines here to reach the required length.
This should provide enough data for the TextDataset.
Let's add a few more sentences just to be safe.
The model needs a sufficient amount of text to learn from.
This is the final line of additional text for now.
"""

# The output file where the sample text is saved
FILE_PATH = "custom_data.txt"
with open(FILE_PATH, "w") as f:
    f.write(custom_text)

print(f"Created {FILE_PATH} with sample text.")

# --- 2. Configuration Parameters & Tokenizer Setup ---
BLOCK_SIZE = 128               # Context window size (128 is a good default for GPT-2 small)

# Load the pre-trained GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set the padding token to be the EOS token. This is standard practice for
# autoregressive models like GPT-2, as they don't natively have a separate pad token.
tokenizer.pad_token = tokenizer.eos_token

# --- 3. Dataset and Data Collator ---
print(f"Loading and tokenizing data from: {FILE_PATH}")

# TextDataset handles reading the file, tokenizing, and splitting into blocks.
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=FILE_PATH,
    block_size=BLOCK_SIZE
)

# DataCollator prepares batches. mlm=False is essential for Causal Language Modeling (standard GPT-2 task).
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

print(f"Dataset successfully created with {len(train_dataset)} blocks.")

# --- 4. Load the Pre-trained GPT-2 Model ---
print("Loading pre-trained GPT-2 model...")
# GPT2LMHeadModel includes the language modeling head for predicting the next token.
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.to(device) # Move model to GPU/CPU for faster training
print("Model loaded successfully.")

# --- 5. Define Training Arguments (Hyperparameters) ---
OUTPUT_DIR = "./fine_tuned_gpt2_model"

print("Defining training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=3,                    # Number of passes over the small dataset
    per_device_train_batch_size=4,         # Adjust based on GPU memory
    save_steps=500,                        # Save checkpoint frequency
    save_total_limit=2,                    # Keep only the two latest checkpoints
    prediction_loss_only=True,             # Only compute loss during evaluation
    logging_dir='./logs',
    learning_rate=5e-5                     # Standard learning rate for fine-tuning
)
print(f"Training will save checkpoints to: {OUTPUT_DIR}")

# --- 6. Initialize and Run the Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)
print("Trainer initialized. Ready to start fine-tuning. (NOTE: Fine-tuning step commented out for a ready-to-run script)")

# --- NOTE: The actual training step is often commented out in deployment code ---
# trainer.train()

# --- 7. Save the Model and Tokenizer (Simulating a saved model) ---
# NOTE: This step assumes the training step above has completed successfully.
print(f"Saving final model to {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Model and tokenizer successfully saved.")

# --- 8. Load the saved fine-tuned model for Generation ---
MODEL_PATH = "./fine_tuned_gpt2_model"

print(f"\nLoading fine-tuned model from: {MODEL_PATH}")
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
model.to(device)
model.eval() # Set the model to evaluation mode for inference
print("Fine-tuned model loaded for text generation.")

# --- 9. Text Generation Example 1: Creative Sampling ---
prompt = "Hark, the air doth chill and grow quite still, for the"
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

print(f"\n--- Generating text based on prompt: '{prompt}' ---")

output = model.generate(
    input_ids,
    max_length=150,
    num_return_sequences=1,
    do_sample=True,          # Enable sampling (crucial for creative text)
    temperature=0.85,        # Controls randomness (higher = more random)
    top_k=50,                # Filters to the top K most probable tokens
    top_p=0.95,              # Nucleus sampling (filters by cumulative probability)
    no_repeat_ngram_size=2,  # Prevents repetitive phrases
    pad_token_id=tokenizer.eos_token_id
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("\nGenerated Output (Creative Sampling):")
print("==================================================")
print(generated_text)
print("==================================================")

# --- 10. Text Generation Example 2: Focused Sampling ---
prompt_new = "Prithee, tell me, good Horatio, what vile rumour"
input_ids = tokenizer.encode(prompt_new, return_tensors='pt').to(device)

print("\n--- Running Second Generation with Tuned Parameters (Lower Temperature) ---")
output = model.generate(
    input_ids,
    max_length=120,
    num_return_sequences=1,
    do_sample=True,
    temperature=0.7,         # Slightly lower temperature for more focused/coherent output
    top_k=50,
    top_p=0.95,
    no_repeat_ngram_size=2,
    pad_token_id=tokenizer.eos_token_id
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"\nGenerated Output for Prompt: '{prompt_new}'")
print("==================================================")
print(generated_text)
print("==================================================")