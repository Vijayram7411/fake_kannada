@echo off
setlocal
cd /d "C:\fakee"

if exist venv\Scripts\activate (
  call venv\Scripts\activate
)

echo Training tuned TF-IDF + Logistic Regression baseline
python src\train_tuned_baselines.py

echo Baseline artifacts saved to models\tfidf_vectorizer.joblib and models\logistic_regression.joblib
