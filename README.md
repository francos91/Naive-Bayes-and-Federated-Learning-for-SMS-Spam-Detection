

#  SMS Spam Detection with Naïve Bayes & Federated Learning

**Author:** F. Saayman  
**Course:** MEng Project
Stellenbosch University  
**Year:** 2026

---

##  Abstract

SMS spam imposes financial and reputational costs on mobile operators, while South Africa's POPIA prohibits centralising raw message content. This work applies CRISP-DM to a synthetic SMS dataset (5,700 messages, 12.3% spam) using Python 3.10. 

A custom Multinomial Naive Bayes classifier is applied to TF-IDF features. Unigrams and bigrams combined with stratified group k-fold (groups = raw message hash) achieves **98.41% accuracy**, **100% spam recall**, and 79 false positives. A horizontal federated simulation using bag-of-words counts and five clients shows that summed per-class counts reproduce the central bag-of-words model exactly (**93.28% accuracy**). 

Adding local differential privacy (ε=0.02) reduces recall to 95.6% while accuracy rises to around 96.7%, illustrating a privacy-utility trade-off. Robustness tests under client dropout (2/5 active) and extreme capacity limits give a mean accuracy of **95.80% (±4.42%)** over ten runs. 

Naive Bayes is highly feasible for federated SMS spam detection on this dataset, but privacy parameters must be tuned to avoid degrading recall.

---

## Key Features

- **Custom Naïve Bayes Implementation** – Built from scratch (no `sklearn.naive_bayes` used for the core logic).
- **CRISP-DM Workflow** – Complete pipeline from Business Understanding to Deployment.
- **Federated Learning Simulation** – Horizontal FL with 5 clients, hash-based grouping to prevent data leakage.
- **Differential Privacy** – Laplace noise injection (ε = 0.02 and ε = 1.0) to simulate local DP constraints.
- **Robustness Testing** – Client dropout (2/5 active) and extreme capacity limits (down to 1% of data per client).
- **Comprehensive EDA** – Word clouds, Zipf's Law, bigram analysis, and metadata correlation heatmaps.

---

## Results Summary

| Model / Configuration | Accuracy | F1-Score (Spam) | Recall (Spam) | FP / FN |
| :--- | :--- | :--- | :--- | :--- |
| **Centralised (Uni+Bi)** | 98.41% | ~0.95 | 100% | 79 FP |
| **Centralised (Unigram Baseline)** | ~93.00% | ~0.85 | ~85% | - |
| **Federated (Non-Private BoW)** | 93.28% | ~0.89 | ~92% | - |
| **Federated + DP (ε = 0.02)** | ~96.70% | ~0.94 | 95.6% | - |
| **Federated + DP (ε = 1.0)** | ~95.80% | ~0.93 | ~93% | - |
| **Heterogeneity (Dropout/Capacity)** | 95.80% (±4.42%) | - | - | - |

---

## Project Structure

```text
.
├── task_b.py                 # Main Python script (contains all logic)
├── synthetic_sms.csv         # Dataset (5,700 synthetic SMS messages)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── A3_OUTPUT/                # Generated outputs (figures, plots)
    ├── class_distribution_pie.png
    ├── length_histogram.png
    ├── boxplot_length.png
    ├── correlation_heatmap.png
    ├── wordclouds.png
    ├── top_10_bigrams.png
    ├── zipf_plot.png
    ├── cm_central_ngrams.png
    ├── heterogeneity_runs.png
    ├── federated_diagram.png
    ├── preprocessing_diagram.png
    └── pipeline_diagram.png
```

---

Requirements

· Python 3.10 or later
· Dependencies (install via pip):

```bash
pip install -r requirements.txt
```

Recommended package versions:

Package Version
numpy ≥ 1.23.0
pandas ≥ 1.5.0
matplotlib ≥ 3.7.0
seaborn ≥ 0.12.0
scipy ≥ 1.10.0
scikit-learn ≥ 1.2.0
wordcloud ≥ 1.8.0

---

How to Run

1. Clone or download this repository and navigate to the folder.
2. Ensure synthetic_sms.csv is in the same directory as the script.
3. Run the script:

```bash
python task_b.py
```

1. The script will:
   · Load and clean the SMS dataset.
   · Perform EDA and save plots to the A3_OUTPUT/ folder.
   · Train and evaluate the centralised custom Naïve Bayes model (5‑fold grouped CV).
   · Simulate federated learning (IID baseline, dropout, capacity limits).
   · Run differential privacy experiments (ε = 0.02 and ε = 1.0).
   · Output final metrics to the terminal.

Note: Random seeds are fixed (e.g., np.random.seed(42) and random_state=112) to ensure full reproducibility.

---

 Methodology Overview

· Data Preparation: Text cleaning (removing special chars, lowercasing), TF-IDF vectorization, and bigram extraction.
· Centralised Model: Custom Multinomial Naïve Bayes with Laplace smoothing (α=1.0). Evaluated using StratifiedGroupKFold to prevent hash-based data leakage.
· Federated Simulation: 5 clients receive non-overlapping data splits. The server aggregates per-class sum counts to compute global conditional probabilities.
· Differential Privacy: Laplace noise (scale = 1/ε) is added to client-side sums before aggregation, demonstrating the privacy-utility trade-off.
· Heterogeneity: Simulates real-world uneven data distribution by dropping clients and reducing sample capacities (1% to 100%).

---

References

· Ballhysa, A. (2025). Global Mobile Messaging Fraud Prevention 2025-30: Trends, Growth. Juniper Research.
· Ballhysa, A. (2026). Global Mobile Messaging Market Report 2026-30: Size, Share, Growth. Juniper Research.
· Bonawitz et al. (2017). Practical secure aggregation for privacy-preserving machine learning. Research.Google, 1175–1191.
· Dala, P. (2017). A framework and model of operation of electronic personal information to achieve and maintain compliance with Condition 7 of the Protection of Personal Information Act. University of Pretoria.
· Dwork, C. (2008). Differential privacy: A survey of results. International Conference on Theory and Applications of Models of Computation, 4978, 1–19.
· Enkono et al. (2020). Application of Machine Learning Classification to Detect Fraudulent E-wallet Deposit Notification SMSes. The African Journal of Information and Communication, 25, 1–12.
· ESET. (2025). Hook, Line, and Sinker: SA’s Phishing Crisis Deepens. Eset.com/Za.
· INTERPOL. (2025). Africa Cyberthreat Assessment Report.
· Juniper Research. (2021). SMS Firewall Revenue to Reach $4.1 Billion Globally by 2026. Business Wire.
· Kandeh, A. T., Botha, R. A., & Futcher, L. A. (2018). Enforcement of the Protection of Personal Information (POPI) Act. SA Journal of Information Management, 20(1).
· Kelleher, J. D., Mac Namee, B., & D'Arcy, A. (2020). Fundamentals of Machine Learning for Predictive Data Analytics (2nd ed.). MIT Press.
· Lin et al. (2022). FedNLP: Benchmarking Federated Learning Methods for Natural Language Processing Tasks. Findings of NAACL 2022, 157–175.
· McCallum, A. (1998). A comparison of event models for naive Bayes text classification.
· Nahapetyan et al. (2023). On SMS phishing tactics and infrastructure. IEEE Symposium on Security and Privacy, 1–16.
· Njoya et al. (2023). Characterizing Mobile Money Phishing Using Reinforcement Learning. IEEE Access, 11, 103839–103862.
· Oyeyemi et al. (2024). SMS Spam Detection and Classification to Combat Abuse in Telephone Networks. Journal of Advances in Mathematics and Computer Science, 38(10), 144–156.
· Pedregosa et al. (2011). Scikit-learn: Machine learning in Python. JMLR, 12, 2825–2830.
· Rose et al. (2024). Next-Gen Phishing Detection System Based on Federated Learning Integrated CNN-LSTM. ICICV 2024, 367–372.
· SA Government. (2013). Protection of Personal Information Act (POPI Act). GCIS.
· Steyn. (2026). This is how much spam to expect on a new SIM card in South Africa. MyBroadband.
· Tang et al. (2022). Clues in tweets: Twitter-guided discovery and analysis of SMS spam. Proceedings of the 2022 ACM SIGSAC Conference, 1, 2751–2764.
· Thuy D. (2025). Global online chat and messaging apps usage 2025. Statista.
· Wen et al. (2022). A survey on federated learning: challenges and applications. International Journal of Machine Learning and Cybernetics, 14(2), 513–535.
· Yurdem et al. (2024). Federated learning: Overview, strategies, applications, tools and future directions. Heliyon, 10(19), e38137.
· Zhang, Y., & Wu, L. (2015). Short text classification based on feature extension using n-gram model. FSKD 2015, 102–106.
· WEF. (2026). Global Cybersecurity Outlook 2026. World Economic Forum.

---

Notes

· This is a proof-of-concept research implementation, not intended for production deployment.
· All experiments are reproducible due to fixed random seeds.
· If you re-run the script, figures in A3_OUTPUT/ will be overwritten. Back them up if needed.

---

© 2026 F. Saayman. 

```
