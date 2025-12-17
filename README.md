🚀 Transfer_Learning_with_Hugging_Face

Fine-tuning DistilBERT for Multi-Class Text Classification on AG News Dataset

🌟 Project Overview

This project demonstrates transfer learning by fine-tuning a DistilBERT model on the AG News dataset to classify news articles into four categories:

Label	Category
0	🌍 World
1	🏀 Sports
2	💼 Business
3	🖥 Sci/Tech

✅ Goal: Use a pre-trained language model for efficient and accurate text classification.

🛠 Key Features

Pre-trained Transformer: DistilBERT (distilbert-base-uncased)

Dataset: AG News via Hugging Face Datasets

Tokenization: AutoTokenizer for converting text into tokens

Dynamic Padding: DataCollatorWithPadding for efficient batching

Training: Hugging Face Trainer API for simplified training and evaluation

Metrics: Accuracy to measure performance

Hub Integration: Optionally push fine-tuned model to Hugging Face Hub

Inference: Use pipeline API for fast predictions

⚡ Installation
pip install transformers datasets evaluate accelerate -U

📝 Usage Example
1. Load Dataset & Tokenize
from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("ag_news")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(row):
    return tokenizer(row["text"], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

2. Load Pre-trained Model
from transformers import AutoModelForSequenceClassification

num_labels = dataset["train"].features["label"].num_classes
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_labels
)

3. Data Collator
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

4. Training Arguments
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="distilbert-ag-news-finetuned",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False
)

5. Define Metrics
import evaluate
import numpy as np

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

6. Train the Model
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

7. Inference Example
from transformers import pipeline

classifier = pipeline(
    "text-classification", 
    model="distilbert-ag-news-finetuned"
)

text = "The home team hit three home runs in the bottom of the ninth to win the game."
prediction = classifier(text)
print(prediction)

# Check label mapping
print(dataset["train"].features["label"].names)

📊 Results

Training Accuracy: ~0.91

Validation Accuracy: ~0.91

Fine-tuned DistilBERT effectively classifies news articles into 4 categories.

💡 Key Learnings

How to perform transfer learning with pre-trained NLP models.

Tokenization, batching, and dynamic padding for efficient training.

Using Hugging Face Trainer for end-to-end model training.

Pushing models to Hugging Face Hub.

Making predictions on real-world text using pipeline API.

📝 License

This project is licensed under the MIT License.

If you want, I can also make a super minimal “one-page portfolio style” README that looks visually stunning for GitHub, with badges, emojis, and fewer code blocks, perfect for recruiters and hiring managers to skim quickly.

Do you want me to make that version too?
