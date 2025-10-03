# Kannada Fake News Detection (Kannada NLP)

This repository contains an end-to-end pipeline to detect fake news in Kannada using NLP.

What’s included
- Data exploration and preprocessing (Kannada-aware)
- Baseline models with TF-IDF + traditional ML (LogReg, SVM, RF, NB)
- BERT-based approach (embeddings + ML)
- Comprehensive evaluation reports and plots
- Streamlit app for interactive inference

Directory structure
- data/: raw and processed data (generated)
- src/: source code (data loading, preprocessing, models, app)
- models/: saved vectorizers, models, and embeddings
- results/: evaluation plots, tables, and final report

Setup
1) Install dependencies
   pip install -r requirements.txt

2) Run the pipeline
   # Explore and save processed dataset
   python src/data_loader.py

   # Preprocess, balance, split
   python src/preprocessor.py

   # Train baseline models (TF-IDF + ML)
   python src/baseline_models.py

   # Train BERT embeddings + ML (optional)
   python src/bert_simple.py

   # Evaluate and generate reports
   python src/model_evaluation.py

Artifacts generated
- results/dataset_overview.png: Dataset distributions
- results/kannada_analysis.png: Kannada text analysis
- results/model_comparison_table.{csv,html}: Metrics summary
- results/comprehensive_model_comparison.png: Performance comparison
- results/all_confusion_matrices.png: Confusion matrices
- results/error_analysis.png: Error breakdown
- results/final_evaluation_report.md: Final report

Best models (on your run)
- TF-IDF + Logistic Regression: Accuracy 0.993, F1 0.993
- BERT + Logistic Regression: Accuracy 0.993, F1 0.993

Run the app
   streamlit run src/app.py

Notes
- GPU recommended for BERT; CPU works for TF-IDF.
- Input is preprocessed to match training settings (Kannada script, stopwords, etc.).
- For production, consider monitoring, periodic retraining, and bias audits.
