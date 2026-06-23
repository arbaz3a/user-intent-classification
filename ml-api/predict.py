"""
predict.py — Core ML prediction module

Loads the pre-trained model artifacts (model, TF-IDF vectorizer, label encoder)
and exposes a predict_intent() function that mirrors the exact same text cleaning
pipeline used during training to ensure consistent predictions.
"""

import re
import os
import numpy as np
import joblib

# ============================================================
# Load model artifacts from the model/ directory
# ============================================================

# Resolve paths relative to this file's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

print(f"[predict.py] Loading model artifacts from: {MODEL_DIR}")

# Load the trained classifier (LinearSVC, LogisticRegression, etc.)
model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
print(f"[predict.py] Model loaded: {type(model).__name__}")

# Load the fitted TF-IDF vectorizer (DO NOT re-fit — use transform only)
tfidf_vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf.pkl"))
print(f"[predict.py] TF-IDF vectorizer loaded (vocab size: {len(tfidf_vectorizer.vocabulary_)})")

# Load the fitted label encoder to map predicted integers back to intent strings
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
print(f"[predict.py] Label encoder loaded ({len(label_encoder.classes_)} classes)")


# ============================================================
# Stopword list — MUST be identical to the training pipeline
# Copied exactly from user_intent_classification.py
# ============================================================

STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
    'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its',
    'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who',
    'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
    'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
    'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
    'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both', 'each',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
    "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
    'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn',
    "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn',
    "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
    'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",
    'won', "won't", 'wouldn', "wouldn't"
}


# ============================================================
# Text cleaning — identical to training pipeline
# ============================================================

def clean_text(text):
    """
    Applies the same 4-step cleaning pipeline as training:
    1. Lowercase
    2. Remove all non-alphabetic characters (keep only a-z and spaces)
    3. Remove stopwords
    4. Strip extra whitespace
    """
    # Step 1: Lowercase
    text = text.lower()

    # Step 2: Remove special characters — keep only lowercase letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)

    # Step 3: Remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS]

    # Step 4: Rejoin and strip
    text = ' '.join(tokens).strip()

    return text


# ============================================================
# Softmax helper for models without predict_proba (e.g. LinearSVC)
# ============================================================

def softmax(x):
    """Compute softmax values for a score array."""
    e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return e_x / e_x.sum()


# ============================================================
# Main prediction function
# ============================================================

def predict_intent(message):
    """
    Predicts the customer support intent for a given message.

    Args:
        message (str): Raw customer support message

    Returns:
        dict: {
            "intent": str,       # Predicted intent label (e.g., "cancel_order")
            "confidence": float  # Confidence score between 0 and 1
        }
    """
    # Step 1: Clean the input using the EXACT same pipeline as training
    cleaned_message = clean_text(message)
    print(f"[predict.py] Original: '{message}' → Cleaned: '{cleaned_message}'")

    # Step 2: Transform using the FITTED TF-IDF vectorizer (NOT fit_transform!)
    message_vector = tfidf_vectorizer.transform([cleaned_message])

    # Step 3: Predict the integer label
    predicted_label_int = model.predict(message_vector)[0]

    # Step 4: Decode the integer label back to human-readable intent string
    predicted_intent = label_encoder.inverse_transform([predicted_label_int])[0]

    # Step 5: Get confidence score
    confidence = 0.0

    if hasattr(model, 'predict_proba'):
        # Models like LogisticRegression, RandomForest have predict_proba
        proba = model.predict_proba(message_vector)[0]
        confidence = float(np.max(proba))
        print(f"[predict.py] Using predict_proba — confidence: {confidence:.4f}")
    elif hasattr(model, 'decision_function'):
        decision_scores = model.decision_function(message_vector)[0]
    
        # Margin based confidence — better than softmax for LinearSVC
        sorted_scores = np.sort(decision_scores)[::-1]
        margin = sorted_scores[0] - sorted_scores[1]
        confidence = float(1 / (1 + np.exp(-margin * 2))) 
        # print(decision_scores)
        # print(probabilities)
        print(f"[predict.py] Margin: {margin:.4f} — confidence: {confidence:.4f}")
    else:
        # Fallback: no confidence available
        confidence = 1.0
        print("[predict.py] No confidence method available — defaulting to 1.0")

    print(f"[predict.py] Predicted intent: '{predicted_intent}' with confidence: {confidence:.4f}")

    return {
        "intent": predicted_intent,
        "confidence": confidence
    }
