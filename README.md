README  Naïve Bayes and Federated Learning for SMS Spam Detection

F. Saayman
----------------------------------------------------------------------------------
Abstract:
-----------------------------------------------------------------------------------
SMS spam imposes financial and reputational costs on mobile operators, while South
Africa’s POPIA prohibits centralising raw message content. This work applies
CRISP-DM to a synthetic SMS dataset (5,700 messages, 12.3% spam) using Python
3.0. A custom Multinomial Naive Bayes classifier is applied to TF-IDF features.
Unigrams and bigrams (representations) combined with stratified group k-fold (groups
= raw message hash) achieves 98.41% accuracy, 100% spam recall, and 79 false
positives. A horizontal federated simulation using bag-of-words counts and five clients
shows that summed per-class counts reproduce the central bag-of-words model
exactly (93.28% accuracy). Adding local differential privacy (ε=0.02) reduces recall to
95.6% while accuracy rises to around 96.7%, illustrating a privacy-utility trade-off.
Robustness tests under client dropout (2/5 active) and extreme capacity limits give a
mean accuracy of 95.80% (±4.42%) over ten runs. Naive Bayes is highly feasible for
federated SMS spam detection on this dataset, but privacy parameters must be tuned
to avoid degrading recall.
----------------------------------------------------------------------------------------------
===============================================================================
1. Contents
===============================================================================

This folder contains the code and resources for Task B:

- `task b.ipynb` – main Google colab notebook OR task b.py file with:
  * data loading and preprocessing (clean_message, digit preservation)
  * centralised Naïve Bayes pipeline (unigram and unigram+bigram TF‑IDF, custom NB)
  * StratifiedGroupKFold (5 folds, groups = raw message hash)
  * federated Naïve Bayes simulation (bag‑of‑words counts, 5 clients, server aggregation)
  * differential privacy experiments (ε = 0.02 and ε = 1.0, Laplace noise) with fixed seed 112
  * robustness heterogeneity experiments (client dropout 2/5, extreme capacity limits)
  * generation of all figures (saved in `A3_OUTPUT/`)

- `synthetic_sms.csv` – synthetic SMS dataset (spam/ham, 5,700 messages) – included.

- `requirements.txt` – list of Python package dependencies (minimal, see Section 2).

- `A3_OUTPUT/` – folder containing generated figures (if notebook is run from scratch):
  * `pipeline_diagram.pdf` – centralised + federated pipeline overview
  * `federated_diagram.pdf` – simplified federated client‑server architecture
  * `preprocessing_diagram.png` – data preparation and feature representations
  * `class_distribution_pie.png`, `length_histogram.png`, `length_boxplot.png` – EDA
  * `top_words_bar_chart.png`, `top_10_bigrams.png`, `zipf_plot.png` – lexical analysis
  * `correlation_heatmap.png` – metadata correlation
  * `wordclouds.png` – spam/ham word clouds
  * `heterogeneity_runs.png` – federated heterogeneity bar chart (10 runs)
  * Additional EDA plots are also saved (confusion matrices, etc.) within the notebook.

===============================================================================
2. Requirements
===============================================================================

- Python 3.10 or later
- Recommended packages (minimal versions):
  * numpy >= 1.23.0
  * pandas >= 1.5.0
  * matplotlib >= 3.7.0
  * seaborn >= 0.12.0
  * scipy >= 1.10.0
  * scikit-learn >= 1.2.0
  * wordcloud >= 1.8.0

The notebook was developed in Google Colab but can be run in any Jupyter environment with these packages installed.

To install all required packages, run:

    pip install -r requirements.txt

(If you prefer to use the exact versions from the Colab environment, run `pip freeze` and filter the list.)

===============================================================================
3. How to run the notebook
===============================================================================

1. Ensure `synthetic_sms.csv` is located in the **same folder** as the notebook.

2. Open `task b.ipynb` in Jupyter, VS Code, or upload to Google Colab.

3. Run all cells from top to bottom. The notebook will:
   - Load and clean the SMS dataset
   - Perform exploratory data analysis (EDA) and save plots to `A3_OUTPUT/`
   - Train and evaluate the centralised custom Naïve Bayes model (5‑fold grouped CV)
   - Simulate federated learning (IID baseline, dropout, capacity limits)
   - Run differential privacy experiments (ε = 0.02 and ε = 1.0)
   - Generate heterogeneity bar chart and final summary.

Random seeds are fixed (e.g., `random_state=112`, `np.random.seed(112)`) to make the results reproducible. The notebook outputs the final metrics exactly as reported in the presentation and report.

===============================================================================
4. Notes
===============================================================================

- This code is intended as a proof‑of‑concept research implementation, not a production system.
- If you re-run the notebook, the saved figures in `A3_OUTPUT/` will be overwritten; ensure you back them up if you need the original outputs.

=============================================================================
5. References
=============================================================================
References:
A.Ballhysa. (2025). Global Mobile Messaging Fraud Prevention 2025-30: Trends,
Growth. Junioer Research. https://www.juniperresearch.com/research/telecomsconnectivity/
messaging/mobile-messaging-fraud-prevention-research-report/
A.Ballhysa. (2026). Global Mobile Messaging Market Report 2026-30: Size, Share,
Growth. Juniper Research. https://www.juniperresearch.com/research/telecomsconnectivity/
messaging/mobile-messaging-research-report/
Bonawitz et al. (2017). Practical secure aggregation for privacy-preserving machine
learning. Research.Google, 1175–1191.
https://doi.org/10.1145/3133956.3133982
Dala, P. (2017). A framework and model of operation for electronic personal
information to achieve and maintain compliance with Condition 7 of the Protection
of Personal [University of Pretoria (South Africa).].
https://search.proquest.com/openview/260cf8fb6e77f7e5eab3ed92cd4fe694/1?p
q-origsite=gscholar&cbl=2026366&diss=y
Dork. (2008). Differential privacy: A survey of results. International Conference on
Theory and Applications of Models of Computation. Berlin, Heidelberg, 4978
LNCS, 1–19. https://doi.org/10.1007/978-3-540-79228-4_1
Enkono et. al. (2020). Application of Machine Learning Classification to Detect
Fraudulent E-wallet Deposit Notification SMSes. The African Journal of
Information and Communication, 25(25), 1–12.
https://doi.org/10.23962/10539/29195
ESET. (2025). Hook, Line, and Sinker: SA’s Phishing Crisis Deepens. Eset.Com/Za.
https://www.eset.com/za/about/newsroom/press-releases-za/pressreleases/
hook-line-and-sinker-sas-phishing-crisisdeepens/?
srsltid=AfmBOoqt6ux46hpvs0gHBz0tGAJaVdmnDedKPSLABqzjhWU1CJf-
VMW
INTERPOL. (2025). Africa Cyberthreat Assessment Report.
https://www.interpol.int/content/download/23094/file/INTERPOL_Africa_Cyberthr
eat_Assessment_Report_2025.pdf
Juniper Research. (2021). Juniper Research: SMS Firewall Revenue to Reach $4.1
Billion Globally by 2026, as Messaging Fraud Evolves. Business Wire.
https://www.proquest.com/docview/2600562002?parentSessionId=Y4hkT%2Bsq
7%2F%2Bs978757tS4XZFIpevM8tcjUhw5N2GJAo%3D&pqorigsite=
primo&searchKeywords=juniper research sms&accountid=14049
Kallner. (2026). SpendTrend26: South Africans pair a cautious spend recovery with a
decisive shift to digital payments. Ebnet. https://www.ebnet.co.za/spendtrend26-
south-africans-pair-a-cautious-spend-recovery-with-a-decisive-shift-to-digitalpayments/
Kandeh, A. T., Botha, R. A., Futcher, L. A., & Kandeh, A. (2018). Enforcement of the
Protection of Personal Information (POPI) Act: Perspective of data management
professionals. SA Journal of Information Management, 20(1).
https://doi.org/10.4102/SAJIM.V20I1.917
Kandeh et. al. (2018). Enforcement of the Protection of Personal Information (POPI)
Act: Perspective of data management professionals. SA Journal of Information
Management, 20(1). https://doi.org/10.4102/SAJIM.V20I1.917
Kelleher, John D.; Brian Mac Namee, A. D. (2020). Fundamentals of Machine Learning
for Predictive Data Analytics, second ed. In MIT press.
https://books.google.co.za/books?hl=en&lr=&id=1Iv-
DwAAQBAJ&oi=fnd&pg=PR15&dq=kelleher+machine+learning+for&ots=263kZv
zTI1&sig=kHWTmRaNsi80G9lPVDowJqbn4Vw&redir_esc=y#v=onepage&q=kell
eher machine learning for&f=false
Lin et al. (2022). FedNLP: Benchmarking Federated Learning Methods for Natural
Language Processing Tasks. Findings of the Association for Computational
Linguistics: NAACL 2022 - Findings, 157–175.
https://doi.org/10.18653/V1/2022.FINDINGS-NAACL.13
McCallum, A. (1998). A comparison of event models for naive bayes text classification.
https://www.academia.edu/download/57514/edb6fzal7cur8b378io.pdf
Nahapetyan et al. (2023). On sms phishing tactics and infrastructure. IEEE Symposium
on Security and Privacy (SP) (Pp. 1-16)., 1–16.
https://ieeexplore.ieee.org/abstract/document/10646609/
Njoya et. al. (2023). Characterizing Mobile Money Phishing Using Reinforcement
Learning. IEEE Access, 11, 103839–103862.
https://doi.org/10.1109/ACCESS.2023.3317692
Oyeyemi et. al. (2024). SMS Spam Detection and Classification to Combat Abuse in
Telephone Networks Using Natural Language Processing. Journal of Advances in
Mathematics and Computer Science, 38(10), 144–156.
https://doi.org/10.9734/JAMCS/2023/v38i101832
Pedregosa et al. (2011). Scikit-learn: Machine learning in Python. Jmlr.Org, 12, 2825–
2830.
http://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf?source=
post_page
Rose et. al. (2024). Next-Gen Phishing Detection System Based on Federated
Learning Integrated CNN-LSTM for SMS Communication. Proceedings - 2024 5th
International Conference on Intelligent Communication Technologies and Virtual
Mobile Networks, ICICV 2024, 367–372.
https://doi.org/10.1109/ICICV62344.2024.00064
SA Government. (2013). Protection of Personal Information Act (POPI Act). GCIS.
https://popia.co.za/
Steyn. (2026). This is how much spam to expect on a new SIM card in South Africa.
MyBroadband. https://mybroadband.co.za/news/cellular/623435-this-is-howmuch-
spam-to-expect-on-a-new-sim-card-in-south-africa.html
Tang et. al. (2022). Clues in tweets: Twitter-guided discovery and analysis of SMS
spam. Proceedings of the 2022 ACM SIGSAC Conference on Computer and
Communications Security, 1, 2751–2764.
https://doi.org/10.1145/3548606.3559351
Thuy D. (2025). Global online chat and messaging apps usage 2025. Statista.
https://www.statista.com/statistics/1489440/chat-and-messenger-serviceusage/?
srsltid=AfmBOorFJ9WdoeXBGMnYsmypC_0S1iO68nHeikoLg2APDoe2
_5ixfqNw
WEF. (2026). Global Cybersecurity Outlook 2026 - Insight Report.
https://reports.weforum.org/docs/WEF_Global_Cybersecurity_Outlook_2026.pdf
Wen et. al. (2022). A survey on federated learning: challenges and applications.
International Journal of Machine Learning and Cybernetics 2022 14:2, 14(2), 513–
535. https://doi.org/10.1007/S13042-022-01647-Y
Yurdem et. al. (2024). Federated learning: Overview, strategies, applications, tools and
future directions. Heliyon, 10(19), e38137.
https://doi.org/10.1016/j.heliyon.2024.e38137
Zhang, Y., & Wu, L. (2015). Short text classification based on feature extension using
n-gram model. In 2015 12th International Conference on Fuzzy Systems and
Knowledge Discovery (FSKD) (pp. 102–106).

===============================================================================
End of README
===============================================================================
