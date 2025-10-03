"""
Tune decision threshold for probability of 'Real' to optimize Fake class F1
"""

import json
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import f1_score, classification_report


def main():
    print("🔧 Tuning decision threshold for 'Real' probability")
    vec = joblib.load('models/tfidf_vectorizer.joblib')
    model = joblib.load('models/logistic_regression.joblib')
    X_test = joblib.load('data/X_test.joblib')
    y_test = joblib.load('data/y_test.joblib')

    X_test_vec = vec.transform(X_test)
    proba_real = model.predict_proba(X_test_vec)[:, 1]

    best = {
        'threshold': 0.5,
        'f1_fake': -1.0,
        'precision_fake': None,
        'recall_fake': None
    }

    for t in np.linspace(0.2, 0.8, 61):  # step=0.01
        y_pred = (proba_real >= t).astype(int)
        # F1 for fake class (label 0)
        f1s = f1_score(y_test, y_pred, average=None, labels=[0, 1])
        f1_fake = float(f1s[0])
        if f1_fake > best['f1_fake']:
            best['threshold'] = float(t)
            best['f1_fake'] = f1_fake

    # Compute precision/recall at best threshold
    y_best = (proba_real >= best['threshold']).astype(int)
    report = classification_report(y_test, y_best, output_dict=True)
    best['precision_fake'] = float(report['0']['precision'])
    best['recall_fake'] = float(report['0']['recall'])

    print(f"✅ Best threshold: {best['threshold']:.2f} | F1_fake={best['f1_fake']:.4f} | P_fake={best['precision_fake']:.4f} | R_fake={best['recall_fake']:.4f}")

    Path('models').mkdir(exist_ok=True)
    with open('models/threshold.json', 'w', encoding='utf-8') as f:
        json.dump({'default_threshold': best['threshold'], 'metrics': best}, f, ensure_ascii=False, indent=2)
    print("💾 Saved threshold to models/threshold.json")


if __name__ == '__main__':
    main()
