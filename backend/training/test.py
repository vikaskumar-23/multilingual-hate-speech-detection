import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the trained model and tokenizer
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_path = os.environ.get("MURIL_MODEL_PATH", "muril_hate_speech_model")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Custom dataset for tokenization
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        inputs = {k: v.squeeze() for k, v in inputs.items()}  # Remove extra dimension
        return inputs

# Function for batch predictions
def classify_text_in_batches(text_list, batch_size=8):
    dataset = TextDataset(text_list, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    all_predictions = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, axis=1)
            all_predictions.extend(predictions.cpu().numpy())
    
    return np.array(all_predictions)

# Load the test data
test_data = pd.read_csv('../preprocessing/test_data.csv')
test_texts = test_data['text'].tolist()
test_labels = test_data['label'].tolist()

# Run predictions on the test set in batches
predictions = classify_text_in_batches(test_texts, batch_size=8)  # Adjust batch_size if still encountering OOM

# Calculate Accuracy, Precision, Recall, F1-Score
accuracy = accuracy_score(test_labels, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predictions, average='binary')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Detailed Classification Report
print("\nClassification Report:")
print(classification_report(test_labels, predictions, target_names=["Non-hate", "Hate"]))

# Per-language F1. An aggregate score hides a model that works in English and
# fails in Marathi - this is the number a reviewer will ask for.
if "lang" in test_data.columns:
    print("\nPer-language results:")
    print(f"{'lang':10} {'n':>7} {'acc':>7} {'prec':>7} {'recall':>7} {'F1':>7}")
    for lang in sorted(test_data["lang"].dropna().unique()):
        mask = (test_data["lang"] == lang).to_numpy()
        if mask.sum() == 0:
            continue
        yt = np.array(test_labels)[mask]
        yp = np.array(predictions)[mask]
        lp, lr, lf1, _ = precision_recall_fscore_support(
            yt, yp, average="binary", zero_division=0)
        print(f"{lang:10} {mask.sum():>7} {accuracy_score(yt, yp):>7.4f} "
              f"{lp:>7.4f} {lr:>7.4f} {lf1:>7.4f}")
else:
    print("\n(no 'lang' column - re-run preprocessing.py to enable per-language metrics)")

# Confusion Matrix
conf_matrix = confusion_matrix(test_labels, predictions)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-hate", "Hate"], yticklabels=["Non-hate", "Hate"])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("\nwrote confusion_matrix.png")
