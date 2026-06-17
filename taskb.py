# -*- coding: utf-8 -*-

# ============================================================
# SMS Spam Detection with Naïve Bayes & Federated Learning
# ============================================================
# MEng
#
# Student: Franco Saayman
#
# Dataset: synthetic_sms.csv (must be placed in the same directory)
#
# Approach:
# Business Understanding → Data Understanding → Data Preparation
# → Modelling → Evaluation → Deployment (CRISP-DM)
# ============================================================

import os, sys, random, warnings, re, math, hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from wordcloud import WordCloud
from matplotlib.patches import FancyBboxPatch

# Configuration
np.random.seed(42)
random.seed(42)
warnings.filterwarnings("ignore")
plt.rcParams['figure.figsize'] = (12, 6)

OUTPUT_DIR = "A3_OUTPUT"
DATA_FILE = "synthetic_sms.csv"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output folder: {OUTPUT_DIR}/")

    # =========================================================================
    # 1. DATA LOADING
    # =========================================================================
    if not os.path.exists(DATA_FILE):
        print(f"Error: '{DATA_FILE}' not found. Please place it in the same directory.")
        sys.exit(1)

    data = []
    with open(DATA_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if ',' in line: label, message = line.split(',', 1)
            elif '\t' in line: label, message = line.split('\t', 1)
            else:
                parts = line.split(' ', 1)
                if len(parts) == 2: label, message = parts
                else: continue
            label = label.strip().lower()
            message = message.strip()
            if label in ['spam', 'ham']:
                data.append((label, message))

    print(f"Loaded {len(data)} messages")
    spam_count = sum(1 for l,_ in data if l=='spam')
    ham_count = len(data) - spam_count
    print(f"Spam: {spam_count}, Ham: {ham_count}")

    df = pd.DataFrame(data, columns=['label', 'message'])
    df['spam'] = df['label'].apply(lambda x: 1 if x == 'spam' else 0)

    # =========================================================================
    # 2. EXPLORATORY DATA ANALYSIS (EDA) & UNDERSTANDING
    # =========================================================================
    print("\n=== Grouped Descriptive Statistics ===")
    print(df.groupby('label').describe())

    total = len(data)
    print(f"\nTotal messages: {total}")
    print(f"Spam: {spam_count} ({spam_count/total*100:.1f}%)")
    print(f"Ham:  {ham_count} ({ham_count/total*100:.1f}%)")
    print(f"Imbalance ratio (ham:spam): {ham_count/spam_count:.1f}:1")

    n_dup_text = df.duplicated(subset=['message']).sum()
    print(f"Duplicate messages (same text): {n_dup_text} ({n_dup_text/len(data)*100:.1f}%)")

    lengths_all = [len(msg) for _, msg in data]
    lengths_spam = [len(msg) for l,msg in data if l=='spam']
    lengths_ham  = [len(msg) for l,msg in data if l=='ham']

    t_stat, p_val = stats.ttest_ind(lengths_spam, lengths_ham)
    print(f"t-test p-value: {p_val:.5f} (significant difference)")

    url_pattern = r'(?:https?://|bit\.ly|tinyurl|wa\.me|ow\.ly|goo\.gl|t\.co)[^\s]+'
    spam_urls = sum(1 for l,msg in data if l=='spam' and re.search(url_pattern, msg, re.I))
    ham_urls  = sum(1 for l,msg in data if l=='ham'  and re.search(url_pattern, msg, re.I))
    print(f"Spam with URLs: {spam_urls}/{spam_count} ({spam_urls/spam_count*100:.1f}%)")
    print(f"Ham with URLs:  {ham_urls}/{ham_count} ({ham_urls/ham_count*100:.1f}%)")

    # Group-by features
    df['length'] = df['message'].apply(len)
    df['num_words'] = df['message'].apply(lambda x: len(x.split()))
    df['num_digits'] = df['message'].apply(lambda x: sum(c.isdigit() for c in x))
    df['num_punct'] = df['message'].apply(lambda x: sum(not (c.isalnum() or c.isspace()) for c in x))
    df['has_url'] = df['message'].str.contains(r'http[s]?://|bit\.ly|tinyurl|wa\.me', regex=True, case=False)
    grouped = df.groupby('label').agg({
        'length': 'mean', 'num_words': 'mean', 'num_digits': 'mean',
        'num_punct': 'mean', 'has_url': 'mean'
    }).round(3)
    print("\n===== Group‑by Comparison (Spam vs Ham) =====")
    print(grouped)

    # VISUALIZATIONS (Saved silently)
    plt.figure(figsize=(6,6))
    plt.pie([ham_count, spam_count], explode=(0,0.1), labels=['Ham', 'Spam'], autopct='%1.1f%%', startangle=90)
    plt.title('Class Distribution')
    plt.savefig(f'{OUTPUT_DIR}/class_distribution_pie.png', dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10,4))
    plt.hist(lengths_spam, bins=30, alpha=0.7, label='Spam', color='red')
    plt.hist(lengths_ham, bins=30, alpha=0.7, label='Ham', color='blue')
    plt.legend()
    plt.title('Message length distribution')
    plt.savefig(f'{OUTPUT_DIR}/length_histogram.png', dpi=150, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 4))
    bp = ax.boxplot([lengths_spam, lengths_ham], labels=['Spam', 'Ham'], patch_artist=True, widths=0.6, showfliers=False)
    bp['boxes'][0].set_facecolor('#ffcccc'); bp['boxes'][1].set_facecolor('#ccccff')
    ax.set_ylabel('Length (characters)'); ax.set_title('Message length distribution')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/boxplot_length.png', dpi=150, bbox_inches='tight')
    plt.close()

    meta_data = [[1 if l=='spam' else 0, len(m), len(m.split())] for l,m in data]
    df_meta = pd.DataFrame(meta_data, columns=['Target (Spam=1)', 'Num_Characters', 'Num_Words'])
    plt.figure(figsize=(7, 5))
    sns.heatmap(df_meta.corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Correlation Heatmap of Message Metadata')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Word clouds
    spam_text = " ".join([msg for l,msg in data if l=='spam'])
    ham_text  = " ".join([msg for l,msg in data if l=='ham'])
    fig, axes = plt.subplots(1,2, figsize=(12,5))
    axes[0].imshow(WordCloud(width=400, height=400).generate(spam_text), interpolation='bilinear')
    axes[0].axis('off'); axes[0].set_title('Spam')
    axes[1].imshow(WordCloud(width=400, height=400).generate(ham_text), interpolation='bilinear')
    axes[1].axis('off'); axes[1].set_title('Ham')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/wordclouds.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Bigrams
    vec = CountVectorizer(ngram_range=(2,2), max_features=20)
    X_bg = vec.fit_transform([msg for l,msg in data if l=='spam'])
    bigrams = vec.get_feature_names_out()
    bigram_freq = sorted(zip(bigrams, X_bg.sum(axis=0).A1), key=lambda x: x[1], reverse=True)[:10]
    plt.figure(figsize=(10,5))
    plt.barh([b for b,_ in bigram_freq], [c for _,c in bigram_freq], color='red')
    plt.gca().invert_yaxis(); plt.title('Top 10 Bigrams in Spam')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/top_10_bigrams.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Zipf's Law
    all_words = [w.lower() for _,msg in data for w in msg.split()]
    sorted_freq = sorted(Counter(all_words).values(), reverse=True)
    ranks = np.arange(1, len(sorted_freq)+1)
    plt.figure(figsize=(8,6))
    plt.loglog(ranks, sorted_freq, 'b.', markersize=2)
    plt.loglog(ranks, sorted_freq[0]/ranks, 'r--', label='Zipf slope -1')
    plt.title("Zipf's Law")
    plt.savefig(f'{OUTPUT_DIR}/zipf_plot.png', dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 3. DATA PREPARATION (Cleaning & Vectorization Functions)
    # =========================================================================
    def clean_message(message):
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', message).lower()
        return re.sub(r'\s+', ' ', cleaned).strip()

    preprocessed_data = [(label, clean_message(msg)) for label, msg in data]

    raw_spam = [msg for label, msg in data if label == 'spam'][:2]
    raw_ham = [msg for label, msg in data if label == 'ham'][:2]
    clean_spam = [msg for label, msg in preprocessed_data if label == 'spam'][:2]
    clean_ham = [msg for label, msg in preprocessed_data if label == 'ham'][:2]
    df_examples = pd.DataFrame({
        'Category': ['spam', 'spam', 'ham', 'ham'],
        'Raw SMS': raw_spam + raw_ham,
        'Cleaned': clean_spam + clean_ham
    })
    print("\nBefore and After Preprocessing Examples:\n", df_examples)

    def build_vocabulary(train_data, min_df=2, max_df=0.9, L=1):
        N = len(train_data)
        doc_freq = defaultdict(int)
        for _, msg in train_data:
            for w in set(msg.split()):
                if len(w) >= L: doc_freq[w] += 1
        return sorted([w for w,f in doc_freq.items() if f >= min_df and f <= max_df*N])

    def compute_tfidf_vectors(train_data, test_data, vocabulary):
        train_counts = [Counter(msg.split()) for _, msg in train_data]
        N = len(train_data)
        doc_freq = {term:0 for term in vocabulary}
        for cnt in train_counts:
            for term in set(cnt.keys()):
                if term in doc_freq: doc_freq[term] += 1
        idf = {term: math.log2((1+N)/(1+doc_freq[term])) + 1 for term in vocabulary}
        def vectorize(cnt):
            total = sum(cnt.values())
            if total == 0: return [0.0]*len(vocabulary)
            return [cnt.get(term,0)/total * idf[term] for term in vocabulary]
        return [vectorize(cnt) for cnt in train_counts], [l for l,_ in train_data], [vectorize(Counter(msg.split())) for _, msg in test_data], [l for l,_ in test_data], idf

    def tfidf_ngrams_dense(train_texts, test_texts, ngram_range=(1,2), min_df=2, max_df=0.9):
        vec = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, max_df=max_df, lowercase=False, token_pattern=r'(?u)\b\w+\b')
        X_tr = vec.fit_transform(train_texts)
        X_te = vec.transform(test_texts)
        return X_tr.toarray().tolist(), X_te.toarray().tolist(), vec.get_feature_names_out().tolist()

    # =========================================================================
    # 4. NAÏVE BAYES LOGIC
    # =========================================================================
    def train_naive_bayes(train_vectors, train_labels, vocabulary):
        X = np.array(train_vectors, dtype=np.float32)
        y = np.array(train_labels)
        n_spam = np.sum(y=='spam'); n_ham = np.sum(y=='ham')
        prior_spam = n_spam/len(y); prior_ham = n_ham/len(y)
        sum_spam = X[y=='spam'].sum(axis=0) if n_spam>0 else np.zeros(len(vocabulary))
        sum_ham  = X[y=='ham'].sum(axis=0)  if n_ham>0  else np.zeros(len(vocabulary))
        V = len(vocabulary)
        prob_spam = (sum_spam+1)/(np.sum(sum_spam)+V)
        prob_ham  = (sum_ham+1)/(np.sum(sum_ham)+V)
        return prior_spam, prior_ham, dict(zip(vocabulary, prob_spam)), dict(zip(vocabulary, prob_ham))

    def naive_bayes_predict(test_vectors, vocabulary, prior_spam, prior_ham, probs_spam, probs_ham):
        preds = []
        for vec in test_vectors:
            log_spam = math.log(prior_spam); log_ham = math.log(prior_ham)
            for term, w in zip(vocabulary, vec):
                if w>0:
                    log_spam += math.log(probs_spam[term])
                    log_ham  += math.log(probs_ham[term])
            preds.append('spam' if log_spam>log_ham else 'ham')
        return preds

    def evaluate_predictions(true_labels, predictions):
        tp = sum(1 for p,t in zip(predictions,true_labels) if p=='spam' and t=='spam')
        fp = sum(1 for p,t in zip(predictions,true_labels) if p=='spam' and t=='ham')
        fn = sum(1 for p,t in zip(predictions,true_labels) if p=='ham' and t=='spam')
        tn = sum(1 for p,t in zip(predictions,true_labels) if p=='ham' and t=='ham')
        acc = (tp+tn)/(tp+tn+fp+fn) if (tp+tn+fp+fn)>0 else 0
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        rec  = tp/(tp+fn) if (tp+fn)>0 else 0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
        return acc, prec, rec, f1, (tp, fp, fn, tn)

    # =========================================================================
    # 5. CENTRALISED MODELING
    # =========================================================================
    y = np.array([l for l,_ in preprocessed_data])
    X_clean = np.array([msg for _,msg in preprocessed_data])
    raw_messages = [msg for _, msg in data]
    group_ids = [hashlib.md5(raw.encode()).hexdigest() for raw in raw_messages]
    unique_groups = {g:i for i,g in enumerate(set(group_ids))}
    groups = np.array([unique_groups[g] for g in group_ids])
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    metrics_ng = {'acc':[],'prec':[],'rec':[],'f1':[]}
    all_true_ng = []; all_pred_ng = []

    print("\n--- Centralised (Unigrams+Bigrams) ---")
    for fold, (train_idx, test_idx) in enumerate(cv.split(X_clean, y, groups)):
        X_tr, X_te, vocab = tfidf_ngrams_dense(X_clean[train_idx].tolist(), X_clean[test_idx].tolist(), (1,2), 2, 0.9)
        prior_s, prior_h, ps, ph = train_naive_bayes(X_tr, y[train_idx].tolist(), vocab)
        y_pred = naive_bayes_predict(X_te, vocab, prior_s, prior_h, ps, ph)
        acc, prec, rec, f1, (tp,fp,fn,tn) = evaluate_predictions(y[test_idx].tolist(), y_pred)
        for m,val in zip(['acc','prec','rec','f1'],[acc,prec,rec,f1]): metrics_ng[m].append(val)
        all_true_ng.extend(y[test_idx].tolist()); all_pred_ng.extend(y_pred)

    cm_ng = confusion_matrix(all_true_ng, all_pred_ng, labels=['ham','spam'])
    ConfusionMatrixDisplay(cm_ng, display_labels=['ham','spam']).plot()
    plt.title("Custom NB (unigrams+bigrams)")
    plt.savefig(f'{OUTPUT_DIR}/cm_central_ngrams.png', dpi=150, bbox_inches='tight')
    plt.close()

    metrics_uni = {'acc':[],'prec':[],'rec':[],'f1':[]}
    all_true_uni = []; all_pred_uni = []

    print("\n--- Centralised (Unigram Baseline) ---")
    for fold, (train_idx, test_idx) in enumerate(cv.split(X_clean, y, groups)):
        train_data_split = [(y[i], X_clean[i]) for i in train_idx]
        test_data_split  = [(y[i], X_clean[i]) for i in test_idx]
        vocab = build_vocabulary(train_data_split, 2, 0.9, 1)
        X_tr, y_tr, X_te, y_te, _ = compute_tfidf_vectors(train_data_split, test_data_split, vocab)
        prior_s, prior_h, ps, ph = train_naive_bayes(X_tr, y_tr, vocab)
        y_pred = naive_bayes_predict(X_te, vocab, prior_s, prior_h, ps, ph)
        acc, prec, rec, f1, (tp,fp,fn,tn) = evaluate_predictions(y_te, y_pred)
        for m,val in zip(['acc','prec','rec','f1'],[acc,prec,rec,f1]): metrics_uni[m].append(val)
        all_true_uni.extend(y_te); all_pred_uni.extend(y_pred)

    # =========================================================================
    # 6. FEDERATED LEARNING SIMULATION
    # =========================================================================
    print("\n--- Federated Learning Simulation ---")
    full_raw = [msg for _, msg in data]
    raw_hashes = [hashlib.md5(r.encode()).hexdigest() for r in full_raw]
    unique_hashes = sorted(list(set(raw_hashes)))

    train_hashes, test_hashes = train_test_split(unique_hashes, test_size=0.2, random_state=112, stratify=None)
    train_idx = [i for i, h in enumerate(raw_hashes) if h in train_hashes]
    test_idx  = [i for i, h in enumerate(raw_hashes) if h in test_hashes]

    X_train_raw = [full_raw[i] for i in train_idx]
    X_test_raw  = [full_raw[i] for i in test_idx]
    y_train_fed = np.array([data[i][0] for i in train_idx])
    y_test_fed  = np.array([data[i][0] for i in test_idx])

    train_cleaned = [(y_train_fed[i], clean_message(X_train_raw[i])) for i in range(len(X_train_raw))]
    global_vocab = build_vocabulary(train_cleaned, min_df=2, max_df=0.9, L=1)

    def to_bow(msg):
        cnt = Counter(clean_message(msg).split())
        return [cnt.get(term, 0) for term in global_vocab]

    X_train_fed = np.array([to_bow(msg) for msg in X_train_raw])
    X_test_fed  = np.array([to_bow(msg) for msg in X_test_raw])

    groups_train = np.array([raw_hashes[i] for i in train_idx])
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    client_splits = list(sgkf.split(X_train_fed, y_train_fed, groups_train))

    client_spam_sums, client_ham_sums, client_spam_cnt, client_ham_cnt = [], [], [], []

    for client_id, (_, client_idx) in enumerate(client_splits, 1):
        Xc, yc = X_train_fed[client_idx], y_train_fed[client_idx]
        spam_mask, ham_mask = (yc == 'spam'), (yc == 'ham')
        spam_cnt, ham_cnt = np.sum(spam_mask), np.sum(ham_mask)
        client_spam_sums.append(np.sum(Xc[spam_mask], axis=0) if spam_cnt > 0 else np.zeros(len(global_vocab)))
        client_ham_sums.append(np.sum(Xc[ham_mask], axis=0) if ham_cnt > 0 else np.zeros(len(global_vocab)))
        client_spam_cnt.append(spam_cnt); client_ham_cnt.append(ham_cnt)

    global_spam_sum = np.sum(client_spam_sums, axis=0)
    global_ham_sum  = np.sum(client_ham_sums, axis=0)
    global_spam, global_ham = sum(client_spam_cnt), sum(client_ham_cnt)

    prior_spam_fed = global_spam / (global_spam + global_ham)
    prior_ham_fed  = global_ham / (global_spam + global_ham)
    alpha, V = 1.0, len(global_vocab)
    cond_spam_fed = (global_spam_sum + alpha) / (global_spam + alpha * V)
    cond_ham_fed  = (global_ham_sum  + alpha) / (global_ham  + alpha * V)

    def predict_federated(vec):
        log_s = np.log(prior_spam_fed) + np.sum(vec * np.log(cond_spam_fed + 1e-9))
        log_h = np.log(prior_ham_fed)  + np.sum(vec * np.log(cond_ham_fed  + 1e-9))
        return 'spam' if log_s > log_h else 'ham'

    y_pred_fed = [predict_federated(v) for v in X_test_fed]

    # ------------------------------------------------------------------
    # Heterogeneity Simulation
    # ------------------------------------------------------------------
    print("\n--- Extreme Heterogeneity (10 Runs) ---")
    accuracies = []
    for run in range(10):
        sgkf_het = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=run)
        c_splits = list(sgkf_het.split(X_train_fed, y_train_fed, groups_train))
        np.random.seed(run)
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
        c_s = (g_ss + alpha) / (g_s + alpha * V); c_h = (g_hs + alpha) / (g_h + alpha * V)

        def pred_het(vec):
            log_s = np.log(p_s) + np.sum(vec * np.log(c_s + 1e-9))
            log_h = np.log(p_h) + np.sum(vec * np.log(c_h + 1e-9))
            return 'spam' if log_s > log_h else 'ham'
        accuracies.append(sum(1 for p, t in zip([pred_het(v) for v in X_test_fed], y_test_fed) if p == t) / len(y_test_fed))

    plt.figure(figsize=(10, 5))
    plt.bar(range(1, 11), accuracies, color='steelblue', edgecolor='black', alpha=0.8)
    plt.axhline(y=np.mean(accuracies), color='red', linestyle='--', label=f'Mean = {np.mean(accuracies):.4f}')
    plt.legend()
    plt.title('Federated Model Accuracy under Heterogeneity (10 runs)')
    plt.savefig(f'{OUTPUT_DIR}/heterogeneity_runs.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ------------------------------------------------------------------
    # Differential Privacy Simulation
    # ------------------------------------------------------------------
    np.random.seed(112)
    def run_dp(epsilon):
        scale = 1.0 / epsilon
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
            log_h = np.log(prior_ham_fed)  + np.sum(vec * np.log(c_h + 1e-9))
            return 'spam' if log_s > log_h else 'ham'
        y_pred_dp = [pred_dp(v) for v in X_test_fed]
        return sum(1 for p, t in zip(y_pred_dp, y_test_fed) if p == t) / len(y_test_fed), y_pred_dp

    acc_dp_02, y_pred_dp_02 = run_dp(0.02)
    acc_dp_10, y_pred_dp_10 = run_dp(1.0)

    # =========================================================================
    # 7. FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*50)
    print("FINAL SUMMARY – ALL METRICS")
    print("="*50)
    print(f"Centralised (unigrams+bigrams) F1: {np.mean(metrics_ng['f1']):.4f} ± {np.std(metrics_ng['f1']):.4f}")
    print(f"Centralised (unigrams baseline) F1: {np.mean(metrics_uni['f1']):.4f} ± {np.std(metrics_uni['f1']):.4f}")
    print(f"Federated non‑private (BoW) F1: {f1_score(y_test_fed, y_pred_fed, pos_label='spam'):.4f}")
    print(f"Federated DP (ε=0.02) F1: {f1_score(y_test_fed, y_pred_dp_02, pos_label='spam'):.4f}")
    print(f"Federated DP (ε=1.0) F1: {f1_score(y_test_fed, y_pred_dp_10, pos_label='spam'):.4f}")

    # =========================================================================
    # 8. ARCHITECTURE DIAGRAMS
    # =========================================================================
    print("\n--- Generating Architecture Diagrams ---")

    def draw_federated_diagram(output_file):
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.patch.set_facecolor('white')
        ax.set_xlim(0, 12); ax.set_ylim(0, 10); ax.axis('off')
        box_style = "round,pad=0.1,rounding_size=0.2"
        ax.add_patch(FancyBboxPatch((3.5, 8.2), 5, 0.8, boxstyle=box_style, facecolor='#e8f5e9', edgecolor='#1f6b3a', linewidth=2, alpha=0.9))
        ax.text(6, 8.6, "Global BoW vocabulary\nBuilt from training set only", ha='center', va='center', fontweight='bold')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

    def draw_diagram(output_file):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis('off')
        ax.text(6, 7.6, "Data Preparation and Feature Representations", ha='center', fontsize=14, fontweight='bold')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

    def draw_professional_diagram(output_file):
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 10))
        fig.patch.set_facecolor('white')
        ax_left.axis('off'); ax_right.axis('off')
        ax_left.set_title("CENTRALISED PIPELINE", fontsize=14, fontweight='bold', pad=25)
        ax_right.set_title("FEDERATED LEARNING SIMULATION", fontsize=14, fontweight='bold', pad=25)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

    draw_federated_diagram(f'{OUTPUT_DIR}/federated_diagram.png')
    draw_diagram(f'{OUTPUT_DIR}/preprocessing_diagram.png')
    draw_professional_diagram(f'{OUTPUT_DIR}/pipeline_diagram.png')
    print("All diagrams saved successfully in A3_OUTPUT.")

if __name__ == "__main__":
    main()
