import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load the trained model and tokenizer
model_name = "distilbert-base-uncased"
model_path = "./results"  # Path to the directory where the trained model is saved
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

# Move model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to predict sentiment of a given text
def predict_sentiment(text):
    # Tokenize the input text
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    
    # Move input tensors to the appropriate device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Make prediction
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Convert logits to predicted label
    predicted_label = torch.argmax(logits, dim=1).item()

    # Map label to sentiment
    sentiment = "positive" if predicted_label == 1 else "negative"
    
    return sentiment

# Example usage
if __name__ == "__main__":
    while True:
        text = input("Enter a text to analyze sentiment (or type 'exit' to quit): ")
        if text.lower() == 'exit':
            break
        sentiment = predict_sentiment(text)
        print(f"Text: {text}\nSentiment: {sentiment}\n")
