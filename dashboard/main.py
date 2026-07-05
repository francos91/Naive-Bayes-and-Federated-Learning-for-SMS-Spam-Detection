# dashboard/main.py
# ============================================================
# FULL SMS SPAM DETECTOR API
# Contains: Custom NB Inference + REAL Federated Learning Sim
# Extracted from task_b.py (PBA3)
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

# --- 1. SUPABASE CONFIG ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("❌ Missing Supabase credentials.")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- 2. FASTAPI APP ---
app = FastAPI(title="SMS Spam Detector", version="1.0.0")

# --- 3. LOAD TRAINED MODEL PARAMETERS (for /predict) ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_params.pkl")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ model_params.pkl not found.")

with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)

vocab = model_data["vocab"]
prior_spam = model_data["prior_spam"]
prior_ham = model_data["prior_ham"]
probs_spam = model_data["probs_spam"]
probs_ham = model_data["probs_ham"]

# --- 4. PREDICTION HELPERS (Inference) ---
def clean_message(message: str) -> str:
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', message).lower()
    return re.sub(r'\s+', ' ', cleaned).strip()

def vectorize_message(message: str):
    words = clean_message(message).split()
    counts = Counter(words)
    total = sum(counts.values())
    if total == 0:
        return [0.0] * len(vocab)
    return [counts.get(term, 0) / total for term in vocab]

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

# --- 5. FEDERATED LEARNING SIMULATION LOGIC (From task_b.py) ---
def build_vocabulary(train_data, min_df=2, max_df=0.9, L=1):
    N = len(train_data)
    doc_freq = defaultdict(int)
    for _, msg in train_data:
        for w in set(msg.split()):
            if len(w) >= L:
                doc_freq[w] += 1
    return sorted([w for w, f in doc_freq.items() if f >= min_df and f <= max_df * N])

def run_federated_simulation(data, epsilon=0.02, runs=10):
    """Full FL + DP + Heterogeneity simulation. Matches task_b.py exactly."""
    # --- Data prep ---
    raw_messages = [msg for _, msg in data]
    raw_hashes = [hashlib.md5(r.encode()).hexdigest() for r in raw_messages]
    unique_hashes = sorted(list(set(raw_hashes)))
    
    train_hashes, test_hashes = train_test_split(
        unique_hashes, test_size=0.2, random_state=112, stratify=None
    )
    train_idx = [i for i, h in enumerate(raw_hashes) if h in train_hashes]
    test_idx = [i for i, h in enumerate(raw_hashes) if h in test_hashes]
    
    X_train_raw = [raw_messages[i] for i in train_idx]
    y_train_fed = np.array([data[i][0] for i in train_idx])
    y_test_fed = np.array([data[i][0] for i in test_idx])
    
    train_cleaned = [(y_train_fed[i], clean_message(X_train_raw[i])) for i in range(len(X_train_raw))]
    global_vocab = build_vocabulary(train_cleaned, min_df=2, max_df=0.9, L=1)
    
    def to_bow(msg):
        cnt = Counter(clean_message(msg).split())
        return [cnt.get(term, 0) for term in global_vocab]
    
    X_train_fed = np.array([to_bow(msg) for msg in X_train_raw])
    X_test_fed = np.array([to_bow(msg) for msg in [raw_messages[i] for i in test_idx]])
    
    groups_train = np.array([raw_hashes[i] for i in train_idx])
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    client_splits = list(sgkf.split(X_train_fed, y_train_fed, groups_train))
    
    # --- Aggregation ---
    client_spam_sums, client_ham_sums, client_spam_cnt, client_ham_cnt = [], [], [], []
    for _, client_idx in client_splits:
        Xc, yc = X_train_fed[client_idx], y_train_fed[client_idx]
        spam_mask, ham_mask = (yc == 'spam'), (yc == 'ham')
        spam_cnt, ham_cnt = np.sum(spam_mask), np.sum(ham_mask)
        client_spam_sums.append(np.sum(Xc[spam_mask], axis=0) if spam_cnt > 0 else np.zeros(len(global_vocab)))
        client_ham_sums.append(np.sum(Xc[ham_mask], axis=0) if ham_cnt > 0 else np.zeros(len(global_vocab)))
        client_spam_cnt.append(spam_cnt); client_ham_cnt.append(ham_cnt)
    
    global_spam_sum = np.sum(client_spam_sums, axis=0)
    global_ham_sum = np.sum(client_ham_sums, axis=0)
    global_spam = sum(client_spam_cnt); global_ham = sum(client_ham_cnt)
    
    alpha, V = 1.0, len(global_vocab)
    prior_spam_fed = global_spam / (global_spam + global_ham)
    prior_ham_fed = global_ham / (global_spam + global_ham)
    cond_spam_fed = (global_spam_sum + alpha) / (global_spam + alpha * V)
    cond_ham_fed = (global_ham_sum + alpha) / (global_ham + alpha * V)
    
    def predict_federated(vec):
        log_s = np.log(prior_spam_fed) + np.sum(vec * np.log(cond_spam_fed + 1e-9))
        log_h = np.log(prior_ham_fed) + np.sum(vec * np.log(cond_ham_fed + 1e-9))
        return 'spam' if log_s > log_h else 'ham'
    
    y_pred_fed = [predict_federated(v) for v in X_test_fed]
    f1_non_private = f1_score(y_test_fed, y_pred_fed, pos_label='spam')
    
    # --- Differential Privacy ---
    def run_dp(eps):
        scale = 1.0 / eps
        css, chs = [], []
        for _, idx in client_splits:
            Xc, yc = X_train_fed[idx], y_train_fed[idx]
            sm, hm = (yc == 'spam'), (yc == 'ham')
            ss = np.sum(Xc[sm], axis=0).astype(float) if np.any(sm) else np.zeros(len(global_vocab), dtype=float)
            hs = np.sum(Xc[hm], axis=0).astype(float) if np.any(hm) else np.zeros(len(global_vocab), dtype=float)
            ss = np.maximum(ss + np.random.laplace(0, scale, size=len(global_vocab)), 0)
            hs = np.maximum(hs + np.random.laplace(0, scale, size=len(global_vocab)), 0)
            css.append(ss); chs.append(hs)
        
        g_ss = np.sum(css, axis=0); g_hs = np.sum(chs, axis=0)
        c_s = (g_ss + alpha) / (global_spam + alpha * V)
        c_h = (g_hs + alpha) / (global_ham + alpha * V)
        
        def pred_dp(vec):
            log_s = np.log(prior_spam_fed) + np.sum(vec * np.log(c_s + 1e-9))
            log_h = np.log(prior_ham_fed) + np.sum(vec * np.log(c_h + 1e-9))
            return 'spam' if log_s > log_h else 'ham'
        
        y_pred_dp = [pred_dp(v) for v in X_test_fed]
        return f1_score(y_test_fed, y_pred_dp, pos_label='spam')
    
    f1_dp_02 = run_dp(0.02)
    f1_dp_10 = run_dp(1.0)
    
    # --- Heterogeneity ---
    np.random.seed(42)
    accuracies = []
    for run_id in range(runs):
        sgkf_het = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=run_id)
        c_splits = list(sgkf_het.split(X_train_fed, y_train_fed, groups_train))
        np.random.seed(run_id)
        active_clients = np.random.choice(range(5), size=2, replace=False)
        
        css, chs, csc, chc = [], [], [], []
        for client_id, (_, client_idx) in enumerate(c_splits, 1):
            if (client_id - 1) not in active_clients: continue
            Xc, yc = X_train_fed[client_idx], y_train_fed[client_idx]
            capacities = [1.0, 0.5, 0.2, 0.05, 0.01]
            sample_frac = capacities[client_id - 1]
            if sample_frac < 1.0:
                n_samples = int(len(Xc) * sample_frac)
                if n_samples == 0: continue
                keep_idx = np.random.choice(len(Xc), n_samples, replace=False)
                Xc, yc = Xc[keep_idx], yc[keep_idx]
            
            sm, hm = (yc == 'spam'), (yc == 'ham')
            sc, hc = np.sum(sm), np.sum(hm)
            css.append(np.sum(Xc[sm], axis=0) if sc > 0 else np.zeros(len(global_vocab)))
            chs.append(np.sum(Xc[hm], axis=0) if hc > 0 else np.zeros(len(global_vocab)))
            csc.append(sc); chc.append(hc)
        
        if len(css) == 0: accuracies.append(0.0); continue
        
        g_ss = np.sum(css, axis=0); g_hs = np.sum(chs, axis=0)
        g_s = sum(csc); g_h = sum(chc)
        p_s = g_s / (g_s + g_h); p_h = g_h / (g_s + g_h)
        c_s = (g_ss + alpha) / (g_s + alpha * V)
        c_h = (g_hs + alpha) / (g_h + alpha * V)
        
        def pred_het(vec):
            log_s = np.log(p_s) + np.sum(vec * np.log(c_s + 1e-9))
            log_h = np.log(p_h) + np.sum(vec * np.log(c_h + 1e-9))
            return 'spam' if log_s > log_h else 'ham'
        
        acc = sum(1 for p, t in zip([pred_het(v) for v in X_test_fed], y_test_fed) if p == t) / len(y_test_fed)
        accuracies.append(acc)
    
    return {
        "f1_non_private": float(f1_non_private),
        "f1_dp_0_02": float(f1_dp_02),
        "f1_dp_1_0": float(f1_dp_10),
        "mean_heterogeneity": float(np.mean(accuracies)),
        "std_heterogeneity": float(np.std(accuracies))
    }

# --- 6. API ENDPOINTS ---
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

@app.post("/simulate")
async def simulate():
    """Runs the REAL Federated Learning + DP simulation using data from Supabase."""
    try:
        # 1. Fetch data from Supabase
        response = supabase.table("sms_messages").select("label", "message").execute()
        data = [(item['label'], item['message']) for item in response.data]
        
        if len(data) == 0:
            raise HTTPException(status_code=404, detail="No data found in Supabase.")
        
        # 2. Run the actual simulation
        metrics = run_federated_simulation(data, epsilon=0.02, runs=10)
        
        return {
            "total_messages": len(data),
            "f1_non_private": metrics["f1_non_private"],
            "f1_dp_0_02": metrics["f1_dp_0_02"],
            "f1_dp_1_0": metrics["f1_dp_1_0"],
            "mean_heterogeneity": metrics["mean_heterogeneity"],
            "std_heterogeneity": metrics["std_heterogeneity"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
