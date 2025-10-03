import streamlit as st
import joblib
import numpy as np
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel

# Local imports
from src.preprocessor import KannadaPreprocessor

st.set_page_config(page_title="Kannada Fake News Detector", page_icon="📰", layout="centered")
# Theming + Header
st.markdown(
    """
    <style>
      .app-header {background: linear-gradient(135deg,#5b86e5 0%,#36d1dc 100%); padding: 22px 18px; border-radius: 14px; color: #fff;}
      .app-title {font-size: 28px; font-weight: 700; margin: 0;}
      .app-sub {opacity: 0.95; margin-top: 6px;}
      .card {background: #ffffff; border: 1px solid rgba(0,0,0,0.06); border-radius: 14px; padding: 16px 16px; box-shadow: 0 6px 18px rgba(0,0,0,0.06);}
      .pill {display: inline-block; padding: 6px 10px; border-radius: 999px; border:1px solid rgba(255,255,255,0.6); margin-top:8px; font-size: 12px;}
      .badge {display:inline-block; padding:4px 8px; border-radius:999px; background:#eef4ff; color:#3b5bdb; margin: 0 6px 6px 0; font-size:12px; border:1px solid #dde6ff}
      .label-fake {background:#ffe8e8; color:#c92a2a; border:1px solid #ffc9c9}
      .label-real {background:#e6fcf5; color:#0c7a5c; border:1px solid #c3fae8}
      .muted {color:#6c757d}
      .divider {height:1px; background:#efefef; margin:10px 0 14px 0}
    </style>
    <div class="app-header">
      <div class="app-title">📰 Kannada Fake News Detector</div>
      <div class="app-sub">Detect Fake (0) vs Real (1) with TF‑IDF / Kannada‑BERT and calibrated thresholds</div>
      <div class="pill">Beta</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load TF-IDF + Logistic Regression
@st.cache_resource(show_spinner=False)
def load_tfidf_pipeline():
    vec_path = Path('models/tfidf_vectorizer.joblib')
    lr_path = Path('models/logistic_regression.joblib')
    if vec_path.exists() and lr_path.exists():
        vectorizer = joblib.load(vec_path)
        lr_model = joblib.load(lr_path)
        return vectorizer, lr_model
    return None, None

# Load BERT (for embeddings) + Logistic Regression (optional)
@st.cache_resource(show_spinner=False)
def load_bert_pipeline():
    model_path = Path('models/bert_bert_logistic_regression.joblib')
    if not model_path.exists():
        return None, None, None
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = AutoTokenizer.from_pretrained('l3cube-pune/kannada-bert')
        bert_model = AutoModel.from_pretrained('l3cube-pune/kannada-bert').to(device)
        lr_model = joblib.load(model_path)
        return tokenizer, bert_model, lr_model
    except Exception:
        return None, None, None

vectorizer, lr_tfidf = load_tfidf_pipeline()
bert_tokenizer, bert_model, lr_bert = load_bert_pipeline()

available_models = []
if vectorizer and lr_tfidf:
    available_models.append('TF-IDF + Logistic Regression')
if Path('models/aug_lr.joblib').exists() and Path('models/aug_word_vectorizer.joblib').exists() and Path('models/aug_char_vectorizer.joblib').exists():
    available_models.append('Augmented TF-IDF (word+char) + LR')
if bert_tokenizer and bert_model and lr_bert:
    available_models.append('BERT + Logistic Regression')

if not available_models:
    st.error("No models found in the models/ directory. Please run training first.")
    st.stop()

model_choice = st.selectbox("Select model", available_models, index=0)

# Load default threshold if available
import json
default_threshold = 0.5
try:
    with open('models/threshold.json', 'r', encoding='utf-8') as f:
        default_threshold = json.load(f).get('default_threshold', 0.5)
except Exception:
    pass

threshold = st.slider("Decision threshold for 'Real' (higher = more conservative)", 0.1, 0.9, float(default_threshold), 0.01)

# Preprocessor (match training settings)
preprocessor = KannadaPreprocessor(
    remove_english=False,
    remove_numbers=True,
    remove_punctuation=True,
    remove_stop_words=True,
    min_length=10,
    max_length=200,
)

left, right = st.columns([1.05, 0.95])
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    text = st.text_area("Input Kannada news text", height=220, placeholder="ಉದಾ: ಸಾಮಾಜಿಕ ಜಾಲತಾಣದಲ್ಲಿ ಹರಡಿದ ಸುಳ್ಳು ವಿಡಿಯೋ ಕ್ಲಿಪ್")
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    model_choice = st.selectbox("Model", available_models, index=0, help="Choose model backend")
    threshold = st.slider("Decision threshold for 'Real'", 0.1, 0.9, float(default_threshold), 0.01, help="Lower = stricter Fake; Higher = stricter Real")
    btn = st.button("Analyze", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Prediction")
    st.markdown('<div class="muted">The model label and confidence appear here after analysis.</div>', unsafe_allow_html=True)
    result_placeholder = st.empty()
    detail_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def embed_with_bert(tokenizer, model, texts, max_length=128):
    device = next(model.parameters()).device
    inputs = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use [CLS] token representation
    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return embeddings

if btn:
    if not text.strip():
        with right:
            result_placeholder.warning("Please enter some text.")
        st.stop()

    cleaned = preprocessor.preprocess_text(text)
    if not cleaned:
        with right:
            result_placeholder.warning("Text became empty after preprocessing. Please enter longer content.")
        st.stop()

    proba = 0.5
    used_model = model_choice
    contributions = None
    cues_found = []

    if model_choice == 'TF-IDF + Logistic Regression':
        X = vectorizer.transform([cleaned])
        proba = float(lr_tfidf.predict_proba(X)[0, 1])
    elif model_choice == 'BERT + Logistic Regression':
        embeddings = embed_with_bert(bert_tokenizer, bert_model, [cleaned])
        if hasattr(lr_bert, 'predict_proba'):
            proba = float(lr_bert.predict_proba(embeddings)[0, 1])
        else:
            score = lr_bert.decision_function(embeddings)
            proba = float(1 / (1 + np.exp(-score))[0])
    else:
        # Augmented TF-IDF model + contributions
        aug_w = joblib.load('models/aug_word_vectorizer.joblib')
        aug_c = joblib.load('models/aug_char_vectorizer.joblib')
        aug_lr = joblib.load('models/aug_lr.joblib')
        Xw = aug_w.transform([cleaned])
        Xc = aug_c.transform([cleaned])
        # custom features in app (replicate training)
        import re, numpy as np
        FAKE_CUE_WORDS = ['ಸುಳ್ಳು','ತಪ್ಪು','ನಕಲಿ','ವದಂತಿ','ಮೋಸ','ನಿಜವಲ್ಲ','ಖಂಡನೆ','ನಿಷೇಧ','ತಿಟ್ಟುಪಡಿ','ಇವತ್ತು ಕರ್ಫ್ಯೂ','ವೈರಲ್','ಎಡಿಟ್','ಹಳೆಯ','AI','ಕೃತಕ','ಗ್ಯಾಜೆಟ್','ಗ್ಯಾಜೆಟ್ ಇಲ್ಲ','ಅಧಿಕೃತ ಇಲ್ಲ','ದಾಖಲೆ ಇಲ್ಲ']
        def custom_one(t):
            count_cues = sum(1 for w in FAKE_CUE_WORDS if w in t)
            years = re.findall(r"\d{4,5}", t)
            imp = 0
            for y in years:
                try:
                    v = int(y)
                    if v >= 2100 or len(y)==5:
                        imp = 1; break
                except: pass
            return np.array([[count_cues, imp]]), [w for w in FAKE_CUE_WORDS if w in t], imp
        Xcust_arr, cues_found, year_flag = custom_one(cleaned)
        from scipy import sparse
        X_all = sparse.hstack([Xw, Xc, sparse.csr_matrix(Xcust_arr)])
        proba = float(aug_lr.predict_proba(X_all)[0,1])
        # contributions
        coef = aug_lr.coef_.ravel()
        n_w = len(aug_w.get_feature_names_out())
        n_c = len(aug_c.get_feature_names_out())
        # word contribs
        w_names = aug_w.get_feature_names_out()
        w_row = Xw.tocoo()
        w_contribs = [(w_names[j], float(v*coef[j])) for j,v in zip(w_row.col, w_row.data)]
        # char contribs
        c_names = aug_c.get_feature_names_out()
        c_row = Xc.tocoo()
        c_contribs = [(c_names[j], float(v*coef[n_w+j])) for j,v in zip(c_row.col, c_row.data)]
        # custom contribs last two
        cust_contribs = [("cues_count", float(Xcust_arr[0,0]*coef[n_w+n_c+0])), ("year_improb", float(Xcust_arr[0,1]*coef[n_w+n_c+1]))]
        all_contribs = w_contribs + c_contribs + cust_contribs
        # sort by absolute impact
        contributions = sorted(all_contribs, key=lambda x: abs(x[1]), reverse=True)[:10]

    # Apply threshold
    pred = 1 if proba >= threshold else 0
    label = 'Real (1)' if pred == 1 else 'Fake (0)'

    with right:
        # Main result
        if pred == 1:
            result_placeholder.markdown(f"<div class='badge label-real'>Prediction: Real (1)</div>", unsafe_allow_html=True)
        else:
            result_placeholder.markdown(f"<div class='badge label-fake'>Prediction: Fake (0)</div>", unsafe_allow_html=True)
        st.progress(float(proba))
        st.caption(f"Confidence P(Real)={proba:.3f} | Threshold={threshold:.2f} | Model={used_model}")
        # Explain
        if contributions:
            st.markdown("#### Top contributing n‑grams")
            cols = st.columns(2)
            with cols[0]:
                for name,val in contributions[:5]:
                    st.markdown(f"<div class='badge'>{name}</div>", unsafe_allow_html=True)
            with cols[1]:
                for name,val in contributions[5:]:
                    st.markdown(f"<div class='badge'>{name}</div>", unsafe_allow_html=True)
        if cues_found:
            st.markdown("#### Cues detected")
            for cue in cues_found:
                st.markdown(f"<span class='badge'>{cue}</span>", unsafe_allow_html=True)

st.info("To run locally: `streamlit run src/app.py`", icon="ℹ️")
