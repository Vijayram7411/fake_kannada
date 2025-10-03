"""
Train a model on the augmented dataset (TF-IDF word+char) with custom cues for fake detection
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, accuracy_score
from scipy import sparse
import re

# Custom feature functions
FAKE_CUE_WORDS = [
    'ಸುಳ್ಳು','ತಪ್ಪು','ನಕಲಿ','ವದಂತಿ','ಮೋಸ','ನಿಜವಲ್ಲ','ಖಂಡನೆ','ನಿಷೇಧ','ತಿದ್ದುಪಡಿ','ಇವತ್ತು ಕರ್ಫ್ಯೂ',
    'ವೈರಲ್','ಎಡಿಟ್','ಹಳೆಯ','AI','ಕೃತಕ','ಗ್ಯಾಜೆಟ್','ಗ್ಯಾಜೆಟ್ ಇಲ್ಲ','ಅಧಿಕೃತ ಇಲ್ಲ','ದಾಖಲೆ ಇಲ್ಲ'
]

def extract_custom_features(texts):
    feats = []
    year_improb = []
    for t in texts:
        t0 = t or ''
        low = t0.lower()
        # count cues
        count_cues = sum(1 for w in FAKE_CUE_WORDS if w in t0)
        # improbable year: any 4+ digit number >= 2100 or weird 5-digit
        years = re.findall(r"\d{4,5}", t0)
        imp = 0
        for y in years:
            try:
                v = int(y)
                if v >= 2100 or len(y) == 5:
                    imp = 1
                    break
            except:
                pass
        year_improb.append(imp)
        feats.append([count_cues])
    A = np.array(feats, dtype=float)
    B = np.array(year_improb, dtype=float).reshape(-1,1)
    return sparse.csr_matrix(np.hstack([A,B]))


def main():
    print("🚀 Training on augmented dataset with custom features")
    df = pd.read_csv('data/augmented_dataset.csv')

    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, stratify=df['label'], random_state=42)

    # Word and char n-grams
    word_vec = TfidfVectorizer(max_features=12000, ngram_range=(1,2), min_df=2, max_df=0.98, lowercase=False)
    char_vec = TfidfVectorizer(analyzer='char', ngram_range=(3,5), min_df=2, max_df=0.98)

    Xw = word_vec.fit_transform(X_train)
    Xc = char_vec.fit_transform(X_train)
    Xcust = extract_custom_features(X_train.tolist())
    X_train_all = sparse.hstack([Xw, Xc, Xcust]).tocsr()

    Xw_te = word_vec.transform(X_test)
    Xc_te = char_vec.transform(X_test)
    Xcust_te = extract_custom_features(X_test.tolist())
    X_test_all = sparse.hstack([Xw_te, Xc_te, Xcust_te]).tocsr()

    # Train LR
    clf = LogisticRegression(max_iter=3000, solver='liblinear')
    grid = GridSearchCV(clf, {'C':[0.5,1.0,2.0,4.0], 'class_weight':[None,'balanced']}, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=0)
    grid.fit(X_train_all, y_train)
    best = grid.best_estimator_
    print('Best params:', grid.best_params_)

    y_pred = best.predict(X_test_all)
    print(classification_report(y_test, y_pred, digits=4))

    # Save artifacts
    Path('models').mkdir(exist_ok=True)
    joblib.dump(word_vec, 'models/aug_word_vectorizer.joblib')
    joblib.dump(char_vec, 'models/aug_char_vectorizer.joblib')
    joblib.dump(best, 'models/aug_lr.joblib')
    print('💾 Saved models/aug_* artifacts')

if __name__ == '__main__':
    main()
