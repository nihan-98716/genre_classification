!pip install --upgrade transformers datasets evaluate -q

# --- Import necessary libraries ---
import pandas as pd
import numpy as np
import torch
import evaluate # The library for calculating metrics
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset

print("✅ Step 1 complete: Libraries installed and imported.")

print("\nDownloading the 'ag_news' dataset from the 'datasets' library...")

# Load the 'ag_news' dataset. It is a standard benchmark and is guaranteed to be available.
dataset = load_dataset("ag_news", split='train')

# Convert the dataset to a pandas DataFrame
df = pd.DataFrame(dataset)

print("\nLarge dataset loaded into a pandas DataFrame.")
print("\n--- Dataset Preview ---")
print(df.head())

print("\n✅ Step 2 complete: Dataset loaded successfully.")


print("\nPreprocessing data...")

final_df = df[['text', 'label']].copy()
final_df.dropna(subset=['text', 'label'], inplace=True) # Drop rows with no text or label
final_df['label'] = final_df['label'].astype(int)

print("Data is in the required format.")
print("\n--- Processed Data Preview ---")
print(final_df.head())
print("\nDistribution of labels (0:World, 1:Sports, 2:Business, 3:Sci/Tech):")
print(final_df['label'].value_counts().sort_index())

print("\n✅ Step 3 complete: Data preprocessed.")


print("\nSplitting data into training and testing sets...")
train_df, test_df = train_test_split(
    final_df,
    test_size=0.2,      # 20% of the data will be used for testing
    random_state=42,    # Ensures the split is the same every time we run
    stratify=final_df['label'] # Ensures train and test sets have similar label distributions
)

print(f"Training set size: {len(train_df)} rows")
print(f"Testing set size: {len(test_df)} rows")

print("\n✅ Step 4 complete: Data split.")


MODEL_NAME = 'roberta-base'
NUM_LABELS = 4 # World, Sports, Business, Sci/Tech

print(f"\nLoading tokenizer and model for '{MODEL_NAME}'...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

print("\n✅ Step 5 complete: Model and tokenizer loaded.")

# First, convert pandas DataFrames back to Hugging Face Dataset objects
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Create a function to tokenize the text
def tokenize_function(examples):
    # Handle potential non-string data in the 'text' column
    texts = [str(x) for x in examples["text"]]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=512)

print("\nTokenizing datasets...")
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

# The datasets are now ready for the trainer.
print("\n✅ Step 6 complete: Datasets tokenized and ready.")


print("\nSetting up training...")

# --- Define the evaluation metric ---
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)


# Define the training arguments.
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,              # 1 epoch is often enough for large datasets
    per_device_train_batch_size=8,   # Use a smaller batch size for larger models
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100, # Log less frequently on large datasets
    report_to="none",
)

# Create the Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    compute_metrics=compute_metrics,
)

# --- Start Training ---
print("\nStarting model training... (This will take 30-60 minutes)")
trainer.train()
print("\n✅ Step 7 complete: Model training finished.")


print("\n--- Making a Prediction ---")

sports_news = "The home team won the championship game in a thrilling overtime victory."
tech_news = "A new AI system can now generate photorealistic images from text descriptions."

# --- Function to make prediction ---
def predict(text):
    print(f"\nInput statement: '{text}'")
    
    device = model.device
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()
    # Updated labels to match the ag_news dataset
    labels = ['World', 'Sports', 'Business', 'Sci/Tech']
    prediction = labels[predicted_class_id]
    print(f"Model Prediction: {prediction}")

    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    confidence = probabilities.max().item() * 100
    print(f"Confidence: {confidence:.2f}%")


# --- Test our statements ---
predict(sports_news)
predict(tech_news)

print("\n✅ Step 8 complete: Prediction made!")

print("\n--- Evaluating Final Model Accuracy ---")

evaluation_results = trainer.evaluate()
accuracy = evaluation_results.get("eval_accuracy")

if accuracy is not None:
    print(f"\nFinal accuracy on the test set: {accuracy * 100:.2f}%")
else:
    print("\nCould not retrieve accuracy. Check evaluation results.")

print("\n✅ Step 9 complete: Evaluation finished.")

print("\n--- Generating Confusion Matrix ---")

# Get predictions for the entire test set
predictions = trainer.predict(tokenized_test_dataset)
predicted_labels = np.argmax(predictions.predictions, axis=1)
true_labels = test_df['label'].tolist()

# Generate the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Define the class names
class_names = ['World', 'Sports', 'Business', 'Sci/Tech']

# Plot the confusion matrix using a heatmap for better visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

print("\n✅ Step 10 complete: Confusion matrix displayed.")
