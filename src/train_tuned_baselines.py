"""
Train tuned baseline models focusing on Fake recall
"""

import joblib
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, f1_score, recall_score, make_scorer
import pandas as pd

from src.preprocessor import KannadaPreprocessor

def main():
    print("🚀 Training tuned Logistic Regression (TF-IDF)")
    X_train = joblib.load('data/X_train.joblib')
    X_test = joblib.load('data/X_test.joblib')
    y_train = joblib.load('data/y_train.joblib')
    y_test = joblib.load('data/y_test.joblib')

    # Vectorizer (more features due to larger set)
    vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1,2), min_df=2, max_df=0.95, lowercase=False)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Tune C and penalty; optimize weighted F1 with emphasis on fake recall
    scorer = {
        'f1_weighted': make_scorer(f1_score, average='weighted'),
        'recall_fake': make_scorer(recall_score, pos_label=0)
    }
    param_grid = {
        'C': [0.25, 0.5, 1.0, 2.0, 4.0],
        'class_weight': [None, 'balanced']
    }
    base = LogisticRegression(max_iter=2000, solver='liblinear')
    grid = GridSearchCV(base, param_grid, scoring=scorer, refit='recall_fake', cv=5, n_jobs=-1, verbose=1)
    grid.fit(X_train_vec, y_train)

    best = grid.best_estimator_
    print(f"✅ Best params: {grid.best_params_}")

    y_pred = best.predict(X_test_vec)
    print(classification_report(y_test, y_pred, digits=4))

    Path('models').mkdir(exist_ok=True)
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.joblib')
    joblib.dump(best, 'models/logistic_regression.joblib')
    print("💾 Saved tuned vectorizer and logistic regression model.")

if __name__ == '__main__':
    main()
