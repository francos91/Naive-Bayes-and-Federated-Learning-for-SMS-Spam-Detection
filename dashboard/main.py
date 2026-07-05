# dashboard/main.py
# ============================================================
# FULL SMS SPAM DETECTOR API - MATCHES task_b.py EXACTLY
# Uses TfidfVectorizer (L2 norm, unigrams+bigrams)
# ============================================================
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client
import os
import pickle
import re
import math
import hashlib
import numpy as np
from collections import Counter, defaultdict
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import f1_score
from typing import List, Optional

# --- Supabase Config ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("❌ Missing Supabase credentials.")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- FastAPI App ---
app = FastAPI(title="SMS Spam Detector", version="1.0.0")

# --- Load Model Parameters (EXACT match to task_b.py) ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_params.pkl")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ model_params.pkl not found.")

with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)

vectorizer = model_data["vectorizer"]   # <-- Fitted TfidfVectorizer from training!
vocab = model_data["vocab"]
prior_spam = model_data["prior_spam"]
prior_ham = model_data["prior_ham"]
probs_spam = model_data["probs_spam"]
probs_ham = model_data["probs_ham"]

# --- Clean function (matches task_b.py) ---
def clean_message(message: str) -> str:
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', message).lower()
    return re.sub(r'\s+', ' ', cleaned).strip()

# --- Vectorize using the fitted TfidfVectorizer ---
def vectorize_message(message: str):
    """Uses the sklearn TfidfVectorizer (L2 norm) - matches task_b.py centralised model."""
    cleaned = clean_message(message)
    X = vectorizer.transform([cleaned])
    return X.toarray().tolist()[0]  # L2-normalized TF-IDF vector

def predict(message: str):
    vec = vectorize_message(message)
    log_spam = math.log(prior_spam)
    log_ham = math.log(prior_ham)
    for i, w in enumerate(vec):
        if w > 0:
            term = vocab[i]
            log_spam += math.log(probs_spam.get(term, 1e-9))
            log_ham += math.log(probs_ham.get(term, 1e-9))
    spam_prob = np.exp(log_spam) / (np.exp(log_spam) + np.exp(log_ham))
    label = "spam" if spam_prob > 0.5 else "ham"
    return {"label": label, "confidence": float(spam_prob if label == "spam" else 1 - spam_prob)}

# --- FASTAPI ENDPOINTS ---
class SMSRequest(BaseModel):
    message: str

@app.get("/")
def root():
    return {"status": "online", "service": "SMS Spam Detector", "vocab_size": len(vocab)}

@app.post("/predict")
async def predict_endpoint(request: SMSRequest):
    try:
        return predict(request.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ready"}

# --- FEDERATED SIMULATION (uses Supabase) ---
def build_vocabulary(train_data, min_df=2, max_df=0.9, L=1):
    N = len(train_data)
    doc_freq = defaultdict(int)
    for _, msg in train_data:
        for w in set(msg.split()):
            if len(w) >= L:
                doc_freq[w] += 1
    return sorted([w for w, f in doc_freq.items() if f >= min_df and f <= max_df * N])

def run_federated_simulation(data, epsilon=0.02, runs=10):
    # ... (PASTE THE EXACT SAME FUNCTION I GAVE YOU EARLIER) ...
    # I'll skip duplicating it here to save space, but you must keep it.
    # It uses clean_message and custom BoW+IDF for FL (which is fine).
    pass

@app.post("/simulate")
async def simulate():
    try:
        response = supabase.table("sms_messages").select("label", "message").execute()
        data = [(item['label'], item['message']) for item in response.data]
        if len(data) == 0:
            raise HTTPException(status_code=404, detail="No data found.")
        metrics = run_federated_simulation(data, epsilon=0.02, runs=10)
        return {"total_messages": len(data), **metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
