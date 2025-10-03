#!/usr/bin/env python3
"""
Generate final evaluation report based on saved results and cross-validation metrics
"""

import json
import os
from pathlib import Path
import joblib
from datetime import datetime

RESULTS_DIR = Path('results')
MODELS_DIR = Path('models')


def load_results():
    results = {}
    # Baseline results
    base_path = MODELS_DIR / 'baseline_results.joblib'
    if base_path.exists():
        try:
            base = joblib.load(base_path)
            results.update(base)
        except Exception:
            pass
    # BERT results
    bert_path = MODELS_DIR / 'bert_results.joblib'
    if bert_path.exists():
        try:
            bert = joblib.load(bert_path)
            results.update(bert)
        except Exception:
            pass
    return results


def summarize_results(all_results):
    summary = []
    for name, metrics in all_results.items():
        summary.append({
            'name': name,
            'accuracy': float(metrics.get('accuracy', 0)),
            'f1': float(metrics.get('f1_score', 0)),
            'precision': float(metrics.get('precision', 0)),
            'recall': float(metrics.get('recall', 0))
        })
    # Sort by F1 desc
    summary.sort(key=lambda x: x['f1'], reverse=True)
    return summary


def load_crossval():
    cv_file = RESULTS_DIR / 'crossval_results.json'
    if cv_file.exists():
        with open(cv_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def render_markdown(summary, cv):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = []
    lines.append('# Kannada Fake News Detection - Comprehensive Model Evaluation Report')
    lines.append('')
    lines.append('## Executive Summary')
    lines.append('')
    if summary:
        top = summary[0]
        lines.append(f"- Best Performing Model: {top['name']}")
        lines.append(f"- Best F1 Score: {top['f1']:.4f}")
        avg_acc = sum(s['accuracy'] for s in summary)/len(summary)
        lines.append(f"- Average Model Accuracy: {avg_acc:.4f}")
        good = sum(1 for s in summary if s['f1'] >= 0.95)
        lines.append(f"- Models with >95% F1 Score: {good}/{len(summary)}")
    lines.append('')
    lines.append('## Model Performance Comparison')
    lines.append('')
    for i, s in enumerate(summary[:10], 1):
        lines.append(f"{i}. {s['name']} - F1: {s['f1']:.4f}, Accuracy: {s['accuracy']:.4f}")
    lines.append('')
    if cv:
        lines.append('## Cross-Validation Summary')
        lines.append('')
        lines.append(f"- Mean Accuracy: {cv.get('mean_accuracy', 0):.4f} (±{cv.get('std_accuracy', 0):.4f})")
        lines.append(f"- Mean F1-Score: {cv.get('mean_f1', 0):.4f} (±{cv.get('std_f1', 0):.4f})")
        lines.append('')
    lines.append('---')
    lines.append(f"*Report generated on: {now}*")
    return '\n'.join(lines)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    all_results = load_results()
    summary = summarize_results(all_results)
    cv = load_crossval()
    md = render_markdown(summary, cv)
    out_path = RESULTS_DIR / 'final_evaluation_report.md'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"Report written to {out_path}")


if __name__ == '__main__':
    main()
