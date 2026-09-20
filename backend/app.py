import os
import re

import requests
import torch
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification

load_dotenv()  # reads backend/.env (gitignored)

app = Flask(__name__)

CORS(app)

# Loaded once at import: model loading is expensive, inference is cheap.
model_path = os.environ.get("MURIL_MODEL_PATH", "training/muril_hate_speech_model")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

from hate_speech_data import HARD_CODED_HATE, HARD_CODED_NON_HATE

# Shared with preprocessing.py so training and serving cannot diverge.
from text_cleaning import clean_text

YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "")

def extract_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if match:
        return match.group(1)
    return url  # Return as-is if already a video ID

def fetch_comments_from_youtube(video_id):
    """Fetch every top-level comment, following nextPageToken."""
    comments = []
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    next_page_token = None

    while True:
        params = {
            'key': YOUTUBE_API_KEY,
            'videoId': video_id,
            'part': 'snippet',
            'maxResults': 100,
            'textFormat': 'plainText',   # no HTML entities / <br> tags
        }
        if next_page_token:
            params['pageToken'] = next_page_token

        data = requests.get(url, params=params, timeout=30).json()
        for item in data.get('items', []):
            comments.append(
                item['snippet']['topLevelComment']['snippet']['textDisplay'])

        next_page_token = data.get('nextPageToken')
        # Cap the crawl so one viral video cannot exhaust the daily quota.
        if not next_page_token or len(comments) >= 500:
            break
    return comments


# Word boundaries, non-hate first: substring matching flagged "I don't hate
# you" and "dumbbell".
def _boundary(phrase):
    return re.compile(r'(?<!\w)' + re.escape(phrase.lower()) + r'(?!\w)')

_NON_HATE_PATTERNS = [_boundary(p) for p in HARD_CODED_NON_HATE]
_HATE_PATTERNS = [_boundary(p) for p in HARD_CODED_HATE]


def lexicon_hint(text):
    """Return 0/1 for an unambiguous match, else None. A hint, not a verdict."""
    low = text.lower()
    for pat in _NON_HATE_PATTERNS:
        if pat.search(low):
            return 0
    for pat in _HATE_PATTERNS:
        if pat.search(low):
            return 1
    return None


def predict_hate_speech(texts):
    """Batched inference. Returns (predictions, confidences)."""
    if not texts:
        return [], []
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128,
                       return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)

    predictions = probs.argmax(dim=1).cpu().numpy()
    confidences = probs.max(dim=1).values.cpu().numpy()
    return predictions, confidences


# Audio support removed: the original transcription path could never run.

# Below this the lexicon breaks the tie; above it the model wins outright.
# 0.70 was too high - it let the lexicon override a correct negation call.
CONFIDENCE_FLOOR = 0.55


def analyze_hate_speech(comments):
    if not comments:
        return 0, [], []

    processed = [clean_text(c) for c in comments]
    preds, confs = predict_hate_speech(processed)   # ONE batched forward pass

    final = []
    for text, pred, conf in zip(processed, preds, confs):
        if conf < CONFIDENCE_FLOOR:
            hint = lexicon_hint(text)               # tiebreak only when unsure
            final.append(int(hint) if hint is not None else int(pred))
        else:
            final.append(int(pred))                 # confident model wins

    hate_speech_percentage = 100.0 * final.count(1) / len(final)

    hate_samples = [c for c, p in zip(comments, final) if p == 1][:5]
    non_hate_samples = [c for c, p in zip(comments, final) if p == 0][:5]

    return hate_speech_percentage, hate_samples, non_hate_samples


@app.route('/analyze', methods=['POST'])
def analyze():
    # Accept JSON or form-encoded.
    data = request.get_json(silent=True) or request.form or {}
    text = data.get('text')
    link = data.get('link')

    if link:
        if "youtube.com" in link or "youtu.be" in link:
            if not YOUTUBE_API_KEY:
                return jsonify({'error': 'YOUTUBE_API_KEY is not set on the server.'}), 500
            video_id = extract_video_id(link)
            comments = fetch_comments_from_youtube(video_id)
            if not comments:
                return jsonify({'error': 'No comments found for this video.'}), 404
            pct, hate_samples, non_hate_samples = analyze_hate_speech(comments)
            return jsonify({
                'hateSpeechPercentage': pct,
                'hateSpeechSamples': hate_samples,
                'nonHateSpeechSamples': non_hate_samples,
                'commentsAnalyzed': len(comments)
            }), 200
        return jsonify({'error': 'Unsupported link. Currently only YouTube links are supported.'}), 400

    if text:
        pct, hate_samples, non_hate_samples = analyze_hate_speech([text])
        return jsonify({
            'hateSpeechPercentage': pct,
            'hateSpeechSamples': hate_samples,
            'nonHateSpeechSamples': non_hate_samples,
            'commentsAnalyzed': 1
        }), 200

    return jsonify({'error': 'Either text or a YouTube link is required'}), 400


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'device': str(device), 'model': model_path}), 200


if __name__ == '__main__':
    # debug=True exposes the Werkzeug console; keep it opt-in. Use gunicorn in prod.
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
