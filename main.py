
import torch
import bz2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification,Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, log_loss, recall_score, precision_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# read in the train and test dataset
train_set= bz2.BZ2File('archive/train.ft.txt.bz2')
test_set = bz2.BZ2File('archive/test.ft.txt.bz2')

def get_labels_and_texts(file, max_samples=1500000):
    labels = []
    texts = []
    count = 0
    for line in bz2.BZ2File(file):
        x = line.decode("utf-8")
        labels.append(int(x[9]) - 1)
        texts.append(x[10:].strip())
        count += 1
        if count >= max_samples:
            break
    return np.array(labels), texts



train_labels, train_texts = get_labels_and_texts('archive/train.ft.txt.bz2',max_samples=1000)
test_labels, test_texts = get_labels_and_texts('archive/test.ft.txt.bz2',max_samples=1000)

train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts,train_labels,test_size=0.2,stratify=train_labels)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

class AmazonDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, max_seq_length, model_name):
        self.texts = texts
        self.labels = labels
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        # Get a single line of text based on index
        text = self.texts[index]
        
        # Get a single line of encoded text
        encoded_text = self.tokenizer.encode_plus(text, 
                                                  add_special_tokens=True, 
                                                  pad_to_max_length=True, 
                                                  max_length=self.max_seq_length,
                                                  truncation=True,
                                                  )
        
        out_dict = {"input_ids" : torch.tensor(encoded_text["input_ids"], dtype=torch.long), 
                    "attention_mask" : torch.tensor(encoded_text["attention_mask"], dtype=torch.long),
                   "label" : torch.tensor(self.labels[index], dtype=torch.long)}
        
        return out_dict
    
    

train_dataset = AmazonDataset(texts=train_texts, labels=train_labels, max_seq_length=256, model_name="distilbert-base-uncased")
val_dataset=AmazonDataset(texts=val_texts, labels=val_labels, max_seq_length=256, model_name="distilbert-base-uncased")
test_dataset=AmazonDataset(texts=test_texts, labels=test_labels, max_seq_length=256, model_name="distilbert-base-uncased")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    corrects = p.label_ids

    accuracy = accuracy_score(corrects, preds)
    log_loss_value = log_loss(corrects, p.predictions)
    recall = recall_score(corrects, preds)
    precision = precision_score(corrects, preds)

    return {
        "accuracy": accuracy,
        "log_loss": log_loss_value,
        "recall": recall,
        "precision": precision,
    }




model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()


# Save the model and tokenizer
model.save_pretrained("./results")
tokenizer.save_pretrained("./results")


# Evaluate the model
eval_result = trainer.evaluate(eval_dataset=test_dataset)

# Print the evaluation results
print(f"Accuracy: {eval_result['eval_accuracy']}")
print(f"Log Loss: {eval_result['eval_log_loss']}")
print(f"Recall: {eval_result['eval_recall']}")
print(f"Precision: {eval_result['eval_precision']}")


