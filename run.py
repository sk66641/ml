from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch

# Load model and tokenizer from the saved folder
model = DistilBertForSequenceClassification.from_pretrained("./promise_claim_model")
tokenizer = DistilBertTokenizerFast.from_pretrained("./promise_claim_model")

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=64)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return "promise" if predicted_class == 1 else "claim"

print(predict("And so our task is to convince people that currently – or have another type of phone to switch, while really taking care of people that have an iPhone so that they choose – when they elect to buy another phone, that they buy another iPhone."))
