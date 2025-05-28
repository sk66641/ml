from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
from data import data  

# Sample Data
# data = [
#     {"text": "I will finish the project by Friday.", "label": 1},  # promise
#     {"text": "She is known to be punctual.", "label": 0},          # claim
#     {"text": "We promise to refund you within 3 days.", "label": 1},
#     {"text": "The company has a good reputation.", "label": 0}
# ]

# Step 1: Convert to Dataset
train_texts, val_texts = train_test_split(data, test_size=0.2, random_state=42)
train_dataset = Dataset.from_list(train_texts)
val_dataset = Dataset.from_list(val_texts)

# Step 2: Tokenization
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding='max_length', max_length=64)

train_dataset = train_dataset.map(tokenize)
val_dataset = val_dataset.map(tokenize)

# Step 3: Model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Step 4: Training
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

trainer.train()
# Save the model and tokenizer
model.save_pretrained("./promise_claim_model")
tokenizer.save_pretrained("./promise_claim_model")
