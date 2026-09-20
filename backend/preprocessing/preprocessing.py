import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

# Shared with app.py so training and serving can never diverge.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from text_cleaning import clean_text

# Load datasets with proper encoding for multi-language support
df_bangla = pd.read_csv('../dataset/bangla.csv', header=0, encoding='utf-8')
df_english = pd.read_csv('../dataset/english.csv', header=0, encoding='utf-8')
df_hindi = pd.read_csv('../dataset/hindi.csv', header=0, encoding='utf-8')
df_marathi = pd.read_csv('../dataset/marathi.csv', header=0, encoding='utf-8')

# Tag language before merging so per-language F1 can be reported later.
for _df, _lang in ((df_marathi, 'marathi'), (df_bangla, 'bangla'),
                   (df_english, 'english'), (df_hindi, 'hindi')):
    _df['lang'] = _lang

# Combine all datasets
df_combined = pd.concat([df_marathi, df_bangla, df_english, df_hindi], ignore_index=True)

# Drop rows with NaN in text or label column
df_combined = df_combined.dropna(subset=['text', 'label'])

# Ensure the label column is integer type
df_combined['label'] = df_combined['label'].astype(int)

# Clean the text data
df_combined['text'] = df_combined['text'].apply(clean_text)

# Remove any rows where text is empty after cleaning
df_combined = df_combined[df_combined['text'] != ""]

# Shuffle data
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the whole frame so lang travels with each row; stratify on label.
y = df_combined['label']
train_data, temp_data = train_test_split(
    df_combined, test_size=0.2, random_state=42, stratify=y)
val_data, test_data = train_test_split(
    temp_data, test_size=0.5, random_state=42, stratify=temp_data['label'])

cols = ['text', 'label', 'lang']
# Save the combined data to CSV files with UTF-8 encoding
train_data[cols].to_csv('train_data.csv', index=False, encoding='utf-8')
val_data[cols].to_csv('val_data.csv', index=False, encoding='utf-8')
test_data[cols].to_csv('test_data.csv', index=False, encoding='utf-8')

print("Data preprocessing and splitting completed. Files saved as 'train_data.csv', 'val_data.csv', and 'test_data.csv'.")
print(f"  train {len(train_data):>6}  val {len(val_data):>6}  test {len(test_data):>6}")
print("  per-language rows:\n" + df_combined['lang'].value_counts().to_string())
