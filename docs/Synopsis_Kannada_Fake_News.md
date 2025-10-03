# Department of Computer Science & Engineering

## SYNOPSIS

1. Title of the Project: Fake News Detection Using Natural Language Processing for Kannada Language
2. Group Number: 67
3. Students:
   - Muhammed Aquif (USN: 4SF22CS114)
   - Pratham Amin (USN: 4SF22CS145)
   - Vishwa (USN: 4SF22CS247)
   - B. M. Shashank (USN: 4SF23CS402)
4. Guide: Mrs. Suketha, Assistant Professor, Dept. of CSE, SCEM

---

## Abstract
The rapid spread of misinformation across digital platforms has made fake news detection a critical challenge. While many systems target English, robust tools for regional Indian languages—especially Kannada—remain limited due to data scarcity, script-specific preprocessing, and code-mixed usage.

This project proposes a Kannada-focused fake news detection system using Natural Language Processing (NLP). We build and evaluate supervised classifiers ranging from TF‑IDF + Logistic Regression to transformer-based Kannada models (e.g., Kannada‑BERT, IndicBERT, and MuRIL) fine‑tuned for binary classification (fake vs. real). Our pipeline covers Kannada‑aware preprocessing (Unicode normalization, tokenization, stop‑word handling), feature extraction (word/character n‑grams and contextual embeddings), model training with class‑imbalance mitigation, threshold calibration to improve fake‑class recall, and an interactive UI for inference.

The expected contributions are: (i) a clean, reproducible Kannada dataset (curated from credible news and fact‑checking sources), (ii) a strong baseline (TF‑IDF + LR) with calibrated thresholds, (iii) a fine‑tuned transformer achieving high F1, and (iv) deployable artifacts (Streamlit app + API). This work demonstrates that language‑specific modeling plus careful calibration significantly improves fake news detection for Kannada content.

**Keywords:** Fake News Detection, Kannada NLP, Text Classification, Transformers, TF‑IDF, Threshold Calibration, Imbalanced Learning

---

## 1. Introduction
The unprecedented growth of social media and messaging platforms has transformed how information is created and consumed. Alongside benefits, misinformation (fake or misleading content) causes public harm—impacting elections, public health, finance, and social trust. Existing solutions mostly target high‑resource languages; Kannada presents unique challenges: script‑aware tokenization, limited labeled data, and frequent code‑mixing.

We address these gaps by building a Kannada‑centric pipeline. Our approach combines:
- Kannada‑aware preprocessing (Unicode normalization, tokenization, stop‑words)
- Feature extraction using TF‑IDF (word + char n‑grams) and transformer embeddings
- Class‑imbalance handling (oversampling and calibrated thresholds)
- Models from linear baselines to fine‑tuned Kannada‑BERT
- An interpretable UI exposing scores, thresholds, and feature cues

This end‑to‑end design aims to make fake‑news detection reliable, efficient, and accessible for Kannada users.

---

## 2. Literature Survey (Selected)
| Ref | Authors | Year | Focus / Contribution |
|-----|---------|------|----------------------|
| [1] | R. Sivanaiah et al. | 2023 | Fake news detection in low‑resource languages; highlights data/tool scarcity and need for tailored pipelines. |
| [2] | S. Sanjana et al. | 2023 | ML, DL, and transformer models for Kannada fake news; shows transformers outperform traditional methods with fine‑tuning. |
| [3] | K. Anirudh et al. | 2024 | Multilingual detection with BERT vs GPT‑3.5; cross‑lingual generalization under data scarcity. |
| [4] | S. S. Nandgaonkar et al. | 2024 | Deep learning for misinformation; emphasizes contextual semantics over rule‑based methods. |
| [5] | E. Raja et al. | 2024 | Hybrid CNN‑BiLSTM for Dravidian languages; improved sequence modeling. |
| [6] | E. Raja et al. | 2024 | Transformer models for Dravidian fake news detection; strong Kannada results after domain fine‑tuning. |
| [7] | S. U. Priya et al. | 2022 | News categorization via ML/DL; compares TF‑IDF/BoW vs embeddings for classification. |
| [8] | S. Singhal et al. | 2021 | Fact‑check factorization for low‑resource Indian languages; data challenges and socio‑linguistic complexity. |
| [9] | A. Agarwal et al. | 2024 | Survey on multilingual deception detection; cross‑lingual robustness and domain shift issues. |
| [10] | K. K. Jayanth et al. | 2023 | XLM‑RoBERTa for Indian languages; improved POS/preprocessing for downstream NLP. |
| [11] | A. Dey et al. | 2023 | Kannada social media dataset analysis; highlights annotation and script complexities. |
| [12] | X. Wang et al. | 2024 | Survey on mono/multilingual misinformation detection for low‑resource languages. |
| [13] | S. K. Suresh, U. Damotharan | 2024 | Kannada‑English code‑mixed synthesis; relevant for understanding code‑mixing effects. |
| [14] | S. Shah et al. | 2020 | Cross‑/multilingual spoken term detection; low‑resource challenges. |
| [15] | M. J. Varma et al. | 2025 | NLP techniques for fake news detection; end‑to‑end pipeline discussion. |

> Note: We curated, standardized, and corrected descriptions for clarity and alignment with the Kannada focus. Full citations appear in References.

---

## 3. Problem Statement
### 3.1 Existing Problem
Most fake‑news detectors underperform on Kannada due to limited labeled data, lack of script‑aware preprocessing, and code‑mixing. This results in poor recall for fake class and user distrust in automated flags.

### 3.2 Proposed Solution
Design a Kannada‑specific, end‑to‑end detector that:
- Curates a clean, labeled Kannada dataset (real from credible publishers; fake from fact‑checks and counter‑claims)
- Implements Kannada‑aware preprocessing
- Trains calibrated models (TF‑IDF + LR; fine‑tuned Kannada‑BERT)
- Exposes interpretable outputs (probabilities, cues, and top n‑grams)
- Deploys as a simple web app/API for real‑time use

---

## 4. Objectives
- Build a robust Kannada NLP pipeline for fake‑news detection
- Create/curate a balanced Kannada dataset with reliable labels
- Implement text preprocessing (normalization, tokenization, stop‑words)
- Train and evaluate baseline (TF‑IDF + LR) and transformer models
- Address class imbalance via oversampling and threshold calibration
- Deliver a deployable app with interpretability and adjustable thresholds

---

## 5. Proposed Methodology
1. **Data Collection**: Aggregate Kannada texts from credible news (real) and fact‑checking sources (fake), remove duplicates, and validate labels.
2. **Preprocessing**: Kannada‑aware normalization; tokenization; stop‑word filtering; handle numerals, URLs, punctuation; optional code‑mix handling.
3. **Feature Extraction**:
   - TF‑IDF word bigrams + character n‑grams (3–5)
   - Contextual embeddings from Kannada‑BERT / IndicBERT / MuRIL
   - Custom cues (e.g., “ಸುಳ್ಳು/ನಕಲಿ/ವೈರಲ್”, improbable years)
4. **Modeling**:
   - Baseline: Logistic Regression (class‑weighted), SVM
   - Advanced: Fine‑tuned Kannada‑BERT (sequence classification)
5. **Imbalance & Calibration**: Oversample minority (fake), tune decision threshold for high fake‑recall without sacrificing real‑precision.
6. **Evaluation**: Accuracy, Precision, Recall, F1 (macro & per‑class), ROC‑AUC; error analysis on misclassified samples.
7. **Deployment**: Streamlit UI and optional REST API; model/artifact versioning.

### 5.1 System Design (Conceptual)
- Preprocessing → Feature Extraction → Classification → Explanation + Thresholding → UI/API

### 5.2 Requirements
#### Functional
- User can submit Kannada text (headline/article) for prediction
- System preprocesses and classifies as Fake/Real; returns probability and cues
- Admin pipeline for periodic retraining and dataset updates

#### Non‑Functional
- Latency: ≤ 2 seconds per short headline on CPU (longer for full articles)
- Robustness: balanced performance across topics and code‑mixing
- Security: sanitized inputs; no storage of PII
- Reliability: ≥ 90% F1 on held‑out Kannada dataset; strong fake‑recall after calibration

---

## 6. Expected Outcome
- A validated, reproducible Kannada fake‑news detector with strong F1 and calibrated fake‑recall
- A curated Kannada dataset with documentation
- A deployable UI exposing predictions, confidence, and cues
- Guidelines for periodic retraining and monitoring

---

## 7. Work Plan (High‑Level)
- Weeks 1–2: Dataset curation and preprocessing
- Weeks 3–4: Baseline modeling (TF‑IDF + LR), calibration, evaluation
- Weeks 5–6: Transformer fine‑tuning (Kannada‑BERT), evaluation
- Weeks 7–8: Deployment (UI/API), documentation, and final report

---

## 8. Conclusion
Language‑specific modeling substantially improves Kannada fake‑news detection. With Kannada‑aware preprocessing, calibrated baselines, and fine‑tuned transformers, we achieve strong per‑class metrics and reliable user‑facing outputs. Remaining challenges include broader domain coverage, robust handling of code‑mixing, and continuous dataset updates to reflect evolving misinformation patterns. Future work includes active learning, multimodal cues (images/video), and human‑in‑the‑loop verification workflows.

---

## References
[1] R. Sivanaiah, N. Ramanathan, S. Hameed, R. Rajagopalan, A. D. Suseelan, M. T. N. Thanagathai, “Fake News Detection in Low‑Resource Languages,” 2023.  
[2] S. Sanjana, S. Kuranagatti, J. G. Devisetti, R. Sharma, A. Arya, “Intersection of ML, DL and Transformers to Combat Fake News in Kannada Language,” 2023.  
[3] K. Anirudh, M. Srikanth, A. Shahina, “Multilingual Fake News Detection in Low‑Resource Languages: A Comparative Study Using BERT and GPT‑3.5,” 2024.  
[4] S. S. Nandgaonkar, J. Shaikh, G. B. Bhore, R. V. Kadam, S. S. Gadhave, “Multilingual Misinformation Detection: Deep Learning Approaches for News Authenticity Assessment,” 2024.  
[5] E. Raja, B. Soni, S. K. Borgohain, “Fake News Detection in Dravidian Languages using Multiscale Residual CNN‑BiLSTM,” 2024.  
[6] E. Raja, B. Soni, S. K. Borgohain, “Fake News Detection in Dravidian Languages Using Transformer Models,” 2024.  
[7] S. U. Priya, S. Shamita, P. B. Honnavali, S. Eswaran, “Multi‑Modal Categorization of News,” 2022.  
[8] S. Singhal, R. R. Shah, P. Kumaraguru, “Factorization of Fact‑Checks for Low‑Resource Indian Languages,” 2021.  
[9] A. Agarwal, Y. P. Singh, V. Rai, “Deciphering Deception: Unmasking Fake News in Multilingual Contexts,” 2024.  
[10] K. K. Jayanth, G. Bharathi Mohan, R. P. Kumar, “Indian Language Analysis with XLM‑RoBERTa,” 2023.  
[11] A. Dey, A. J. Aishwaryasri, J. Surya, J. Mg, P. Kannadaguli, “Exploring Social Media Trends – A Kannada Dataset Analysis,” 2023.  
[12] X. Wang, W. Zhang, S. Rajtmajer, “Monolingual and Multilingual Misinformation Detection for Low‑Resource Languages: A Survey,” 2024.  
[13] S. K. Suresh, U. Damotharan, “Kannada‑English Code‑Mixed Speech Synthesis,” 2024.  
[14] S. Shah, S. Guha, S. Khanuja, S. Sitaram, “Cross‑lingual and Multilingual Spoken Term Detection,” 2020.  
[15] M. J. Varma, M. S. Rohit, G. S. G. Selvi, “Fake News Detection using NLP,” 2025.
