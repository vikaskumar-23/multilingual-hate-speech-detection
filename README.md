# Multilingual Hate Speech Detection

![Project Screenshot](/screenshot/screenshot.png)

## Overview
This project is a **Multilingual Hate Speech Detection System** that detects hate speech in **English, Hindi, Bengali and Marathi**. It supports **text input, and YouTube comment detection**. The system is built using a **fine-tuned MuRIL model**, a **Flask backend**, and a **React + Tailwind frontend**.

## Features
- **Multilingual Text Detection**: Supports multiple Indian languages.
- **Audio Hate Speech Detection**: Converts speech to text and detects hate speech.
- **YouTube Comment Analysis**: Extracts comments from YouTube videos and classifies them.
- **Real-Time Processing**: Provides instant results for user inputs.
- **Web-Based Interface**: User-friendly UI built with React and Tailwind CSS.

## Tech Stack
- **Frontend**: React, Tailwind CSS
- **Backend**: Flask, FastAPI
- **Model**: MuRIL (Fine-tuned for hate speech detection)
- **Database**: MongoDB / Firebase (optional for logging results)
- **APIs & Libraries**: YouTube API, Speech-to-Text (Google API, Whisper, or similar), Transformers, Torch, Scikit-learn

## Results

Fine-tuned `google/muril-base-cased` for 3 epochs (12,714 steps, batch 16, lr 2e-5,
fp16) on an RTX 3060. Evaluated on the **held-out test split: 8,476 comments**
never seen during training. Reproduce with `python training/test.py`.

### Overall

| Metric | This model | Published baseline* |
|---|---|---|
| Accuracy | **0.8541** | 0.8043 |
| Precision | 0.8068 | 0.8097 |
| Recall | **0.8736** | 0.7189 |
| **F1** | **0.8389** | 0.7616 |

\* `Hate-speech-CNERG/indic-abusive-allInOne-MuRIL`, scored on the identical test split.

Precision is effectively tied; the gain is almost entirely **recall** — the baseline
misses 28% of hate speech, this model misses 13%. For a moderation tool that is the
axis that matters, since a missed slur harms someone while a false positive is
appealable.

### Per language

| Language | n | Accuracy | Precision | Recall | F1 | Baseline F1 |
|---|---|---|---|---|---|---|
| Marathi | 3,078 | 0.9068 | 0.9015 | 0.9126 | **0.9070** | 0.8104 |
| Bengali | 2,934 | 0.8947 | 0.8162 | 0.8934 | 0.8531 | **0.9033** |
| English | 1,508 | 0.8322 | 0.7952 | 0.8865 | **0.8383** | 0.6254 |
| Hindi | 956 | 0.5941 | 0.5195 | 0.6544 | **0.5792** | 0.4436 |

### Reading these honestly

**Hindi is weak — 0.58 F1, barely above chance.** The aggregate 0.84 is carried by
Marathi and Bengali. Hindi is also the smallest split (956 test rows, ~11% of the
corpus) and the one where spot-checking turned up mislabelled rows — a
misinformation example tagged as hate. More data and a relabelling pass are the
obvious next steps, not more training.

**Bengali is the one place the baseline wins**, by 5 F1 points.

**The comparison favours this model by construction.** The test split is drawn from
the same corpus it trained on, so it shares annotation conventions and distribution.
The baseline was trained elsewhere on ten languages. The fair claim is "better on
this distribution", not "better in general".

Reporting per-language numbers at all is the point: a single aggregate would have
hidden the Hindi result entirely.

### Confusion matrices

- `training/confusion_matrix.png` — this model
- `training/confusion_matrix_cnerg_baseline.png` — published baseline

## Installation
### Prerequisites
- Python 3.9+
- Node.js 18+
- GPU with CUDA support (recommended for faster inference)

### Backend Setup
```bash
# Clone the repository
git clone https://github.com/vikaskumar-23/multilingual-hate-speech-detection.git
cd multilingual-hate-speech-detection/backend

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the backend
python app.py
```

### Frontend Setup
```bash
cd ../frontend

# Install dependencies
npm install

# Start the frontend
npm start
```

## Usage
1. **Text Analysis**: Enter text and check if it contains hate speech.
2. **Audio Detection**: Upload an audio file or use live speech detection.
3. **YouTube Analysis**: Enter a video link to analyze its comments.
4. **Real-time Results**: Get an immediate classification output.

## Model Files
Since the model files and results folder have been removed due to their large size, you can download the pre-trained model from:
[MuRIL Fine-tuned Model on Hugging Face](https://huggingface.co/Hate-speech-CNERG/indic-abusive-allInOne-MuRIL/tree/main)

## Model Training & Fine-tuning
1. Prepare your dataset (`train_dataset.pt`, `val_dataset.pt`).
2. Run the training script:
   ```bash
   python train.py --epochs 3 --batch_size 16 --lr 5e-5
   ```
3. Save and deploy the trained model for inference.
