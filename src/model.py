import json
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from seqeval.metrics import classification_report
import numpy as np
# fine tuning bert pipeline

model_name = "nlpaueb/bert-base-greek-uncased-v1"

# Aspect term extraction and token classification
label_map = {"O": 0, "B-ASP": 1, "I-ASP": 2}
num_labels = len(label_map)
   
# Function to load and preprocess the dataset
def load_dataset(file_path, tokenizer, label_map):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert to a format compatible with the datasets library
    formatted_data = []
    for item in data:
        tokens = item['tokens']
        labels = item['bio_labels']
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        label_ids = [label_map.get(label, -100) for label in labels]
        formatted_data.append({'input_ids': input_ids, 'labels': label_ids})

    return Dataset.from_list(formatted_data)


model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=num_labels)

# model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load datasets
train_dataset_path = "data/processed_data/processed_aspect_data_train.json"
val_dataset_path = "data/processed_data/processed_aspect_data_val.json"

train_dataset = load_dataset(train_dataset_path, tokenizer, label_map)
val_dataset = load_dataset(val_dataset_path, tokenizer, label_map)


# Define compute_metrics function
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_map_inverse[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_map_inverse[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = classification_report(true_labels, true_predictions, output_dict=True)
    return {
        "precision": results["macro avg"]["precision"],
        "recall": results["macro avg"]["recall"],
        "f1": results["macro avg"]["f1"],
        "accuracy": results["accuracy"],
    }

# Add inverse label map for decoding
label_map_inverse = {v: k for k, v in label_map.items()}

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="steps",
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# trainer.train()
trainer.train()

# Save the trained model and tokenizer
model.save_pretrained("aspect_extractor_model")
tokenizer.save_pretrained("aspect_extractor_model")
