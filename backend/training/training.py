import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import (accuracy_score,
                             precision_recall_fscore_support)
import numpy as np
import pandas as pd


def compute_metrics(eval_pred):
    """Without this the Trainer reports only loss - never accuracy or F1."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {'accuracy': accuracy_score(labels, preds),
            'precision': p, 'recall': r, 'f1': f1}

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the MuRIL tokenizer and model
model_name = "google/muril-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

# Load the preprocessed data
train_data = pd.read_csv('../preprocessing/train_data.csv')
val_data = pd.read_csv('../preprocessing/val_data.csv')
test_data = pd.read_csv('../preprocessing/test_data.csv')

# Convert to Hugging Face dataset format
train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)
test_dataset = Dataset.from_pandas(test_data)

# Tokenize the datasets
def tokenize(batch):
    # padding=False: DataCollatorWithPadding pads per batch instead
    return tokenizer(batch['text'], padding=False, truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# Set format for PyTorch
train_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'label'])

# Define training arguments with matching save and evaluation strategy
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",       # Set evaluation to occur every epoch
    save_strategy="epoch",             # Set saving to occur every epoch
    learning_rate=2e-5,
    per_device_train_batch_size=16,    # Increased batch size
    per_device_eval_batch_size=16,     # Increased batch size for evaluation
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1",        # select on F1, not loss
    greater_is_better=True,
    # ~2x faster on Ampere, halves activation memory. No effect on CPU.
    fp16=torch.cuda.is_available(),
    dataloader_pin_memory=torch.cuda.is_available(),
    report_to=["tensorboard"],
)



# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()

# Evaluate the model on the test set
test_results = trainer.evaluate(test_dataset)
print("Test Results:", test_results)

# Save the model and tokenizer
model.save_pretrained("muril_hate_speech_model")
tokenizer.save_pretrained("muril_hate_speech_model")


