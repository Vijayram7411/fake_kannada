"""
Comprehensive Model Evaluation and Comparison
============================================

This module provides comprehensive evaluation and comparison of all models
trained for Kannada fake news detection, including baseline TF-IDF models
and BERT-based approaches.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ModelEvaluator:
    """
    Comprehensive model evaluation and comparison class.
    """
    
    def __init__(self, results_dir='models', output_dir='results'):
        """
        Initialize the evaluator.
        
        Parameters:
        -----------
        results_dir : str
            Directory containing model results
        output_dir : str
            Directory to save evaluation outputs
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Storage for all model results
        self.all_results = {}
        self.test_data = None
        
        print("🔍 Initialized Model Evaluator")
        print(f"   - Results directory: {results_dir}")
        print(f"   - Output directory: {output_dir}")
    
    def load_all_results(self):
        """Load all model results from saved files."""
        print("📂 Loading all model results...")
        
        # Load baseline results
        baseline_path = self.results_dir / 'baseline_results.joblib'
        if baseline_path.exists():
            baseline_results = joblib.load(baseline_path)
            self.all_results.update(baseline_results)
            print(f"   ✅ Loaded {len(baseline_results)} baseline models")
        
        # Load BERT results
        bert_path = self.results_dir / 'bert_results.joblib'
        if bert_path.exists():
            bert_results = joblib.load(bert_path)
            self.all_results.update(bert_results)
            print(f"   ✅ Loaded {len(bert_results)} BERT models")
        
        # Load test data
        try:
            y_test = joblib.load('data/y_test.joblib')
            X_test = joblib.load('data/X_test.joblib')
            self.test_data = {'X_test': X_test, 'y_test': y_test}
            print(f"   ✅ Loaded test data: {len(y_test)} samples")
        except:
            print("   ⚠️ Could not load test data")
        
        print(f"📊 Total models loaded: {len(self.all_results)}")
        return self.all_results
    
    def create_comprehensive_comparison(self):
        """Create comprehensive comparison of all models."""
        print("📊 Creating comprehensive model comparison...")
        
        if not self.all_results:
            print("❌ No results loaded. Please run load_all_results() first.")
            return
        
        # Prepare comparison data
        comparison_data = []
        
        for model_name, results in self.all_results.items():
            # Extract metrics
            accuracy = results.get('accuracy', 0)
            precision = results.get('precision', 0)
            recall = results.get('recall', 0)
            f1 = results.get('f1_score', 0)
            auc = results.get('auc', 0)
            
            # Determine model type
            if 'BERT' in model_name:
                model_type = 'BERT-based'
                complexity = 'High'
            else:
                model_type = 'Traditional ML'
                complexity = 'Low'
            
            comparison_data.append({
                'Model': model_name,
                'Type': model_type,
                'Complexity': complexity,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'AUC': auc
            })
        
        # Create DataFrame
        df_comparison = pd.DataFrame(comparison_data)
        
        # Create comprehensive visualization
        self._plot_comprehensive_comparison(df_comparison)
        
        # Create detailed metrics table
        self._create_metrics_table(df_comparison)
        
        return df_comparison
    
    def _plot_comprehensive_comparison(self, df_comparison):
        """Create comprehensive comparison plots."""
        # Create subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Overall performance comparison
        ax1 = plt.subplot(2, 3, 1)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
        
        x_pos = np.arange(len(df_comparison))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            offset = (i - 2) * width
            bars = ax1.bar(x_pos + offset, df_comparison[metric], width, 
                          label=metric, alpha=0.8)
            
            # Add value labels on bars
            for bar, value in zip(bars, df_comparison[metric]):
                if value > 0:  # Only label non-zero values
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                            f'{value:.3f}', ha='center', va='bottom', fontsize=8, rotation=45)
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Score')
        ax1.set_title('All Models Performance Comparison', fontweight='bold', fontsize=14)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([name.replace(' ', '\\n') for name in df_comparison['Model']], 
                           rotation=45, ha='right', fontsize=10)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.set_ylim(0, 1.1)
        
        # 2. Model type comparison
        ax2 = plt.subplot(2, 3, 2)
        type_performance = df_comparison.groupby('Type')[['Accuracy', 'F1 Score']].mean()
        type_performance.plot(kind='bar', ax=ax2, alpha=0.8)
        ax2.set_title('Performance by Model Type', fontweight='bold')
        ax2.set_ylabel('Average Score')
        ax2.set_xticklabels(type_performance.index, rotation=0)
        ax2.legend()
        
        # Add value labels
        for container in ax2.containers:
            ax2.bar_label(container, fmt='%.3f', rotation=0)
        
        # 3. F1 Score ranking
        ax3 = plt.subplot(2, 3, 3)
        df_sorted = df_comparison.sort_values('F1 Score', ascending=True)
        colors = ['red' if f1 < 0.9 else 'orange' if f1 < 0.95 else 'green' 
                 for f1 in df_sorted['F1 Score']]
        
        bars = ax3.barh(range(len(df_sorted)), df_sorted['F1 Score'], color=colors, alpha=0.7)
        ax3.set_yticks(range(len(df_sorted)))
        ax3.set_yticklabels([name.replace(' ', '\\n') for name in df_sorted['Model']], fontsize=10)
        ax3.set_xlabel('F1 Score')
        ax3.set_title('F1 Score Ranking', fontweight='bold')
        ax3.set_xlim(0, 1)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, df_sorted['F1 Score'])):
            width = bar.get_width()
            ax3.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{value:.4f}', ha='left', va='center', fontweight='bold')
        
        # 4. Accuracy vs F1 Score scatter
        ax4 = plt.subplot(2, 3, 4)
        colors_dict = {'Traditional ML': 'blue', 'BERT-based': 'red'}
        
        for model_type in df_comparison['Type'].unique():
            subset = df_comparison[df_comparison['Type'] == model_type]
            ax4.scatter(subset['Accuracy'], subset['F1 Score'], 
                       c=colors_dict[model_type], label=model_type, s=100, alpha=0.7)
            
            # Add model name annotations
            for idx, row in subset.iterrows():
                ax4.annotate(row['Model'].replace(' ', '\\n'), 
                           (row['Accuracy'], row['F1 Score']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax4.set_xlabel('Accuracy')
        ax4.set_ylabel('F1 Score')
        ax4.set_title('Accuracy vs F1 Score', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Set equal aspect ratio for better visualization
        ax4.set_xlim(0.85, 1.0)
        ax4.set_ylim(0.85, 1.0)
        
        # 5. Performance distribution
        ax5 = plt.subplot(2, 3, 5)
        metrics_melted = df_comparison[['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score']].melt(
            id_vars='Model', var_name='Metric', value_name='Score'
        )
        
        sns.boxplot(data=metrics_melted, x='Metric', y='Score', ax=ax5)
        ax5.set_title('Performance Distribution Across Metrics', fontweight='bold')
        ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45)
        
        # 6. Top 3 models detailed comparison
        ax6 = plt.subplot(2, 3, 6)
        top_3 = df_comparison.nlargest(3, 'F1 Score')
        
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        x = np.arange(len(metrics_to_plot))
        width = 0.25
        
        for i, (idx, model) in enumerate(top_3.iterrows()):
            offset = (i - 1) * width
            values = [model[metric] for metric in metrics_to_plot]
            bars = ax6.bar(x + offset, values, width, label=model['Model'], alpha=0.8)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax6.set_xlabel('Metrics')
        ax6.set_ylabel('Score')
        ax6.set_title('Top 3 Models Detailed Comparison', fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels(metrics_to_plot)
        ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax6.set_ylim(0.9, 1.0)
        
        plt.tight_layout()
        
        # Save the plot
        save_path = self.output_dir / 'comprehensive_model_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Comprehensive comparison saved to {save_path}")
        plt.show()
    
    def _create_metrics_table(self, df_comparison):
        """Create and save detailed metrics table."""
        print("📋 Creating detailed metrics table...")
        
        # Sort by F1 Score
        df_sorted = df_comparison.sort_values('F1 Score', ascending=False)
        
        # Add ranking
        df_sorted['Rank'] = range(1, len(df_sorted) + 1)
        
        # Reorder columns
        columns = ['Rank', 'Model', 'Type', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
        df_final = df_sorted[columns].copy()
        
        # Round numerical values
        numerical_cols = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
        df_final[numerical_cols] = df_final[numerical_cols].round(4)
        
        # Save to CSV
        csv_path = self.output_dir / 'model_comparison_table.csv'
        df_final.to_csv(csv_path, index=False)
        
        # Create a nice HTML table
        html_table = df_final.to_html(index=False, classes='table table-striped table-bordered',
                                     table_id='model-comparison')
        
        # Add some basic styling
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Kannada Fake News Detection - Model Comparison</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; text-align: center; }}
                .table {{ margin: 20px auto; border-collapse: collapse; width: 100%; }}
                .table th {{ background-color: #4CAF50; color: white; padding: 12px; text-align: center; }}
                .table td {{ padding: 8px 12px; text-align: center; border: 1px solid #ddd; }}
                .table tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .rank-1 {{ background-color: #FFD700 !important; font-weight: bold; }}
                .rank-2 {{ background-color: #C0C0C0 !important; font-weight: bold; }}
                .rank-3 {{ background-color: #CD7F32 !important; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>Kannada Fake News Detection - Model Performance Comparison</h1>
            <p style="text-align: center; color: #666;">
                Comprehensive evaluation of all models trained on the Kannada fake news dataset
            </p>
            {html_table}
            <p style="text-align: center; margin-top: 30px; color: #888; font-size: 12px;">
                Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
            </p>
        </body>
        </html>
        """
        
        # Save HTML table
        html_path = self.output_dir / 'model_comparison_table.html'
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📋 Metrics table saved to:")
        print(f"   - CSV: {csv_path}")
        print(f"   - HTML: {html_path}")
        
        # Display top 3 models
        print(f"\\n🏆 TOP 3 MODELS:")
        print("=" * 60)
        for idx, row in df_final.head(3).iterrows():
            print(f"{row['Rank']}. {row['Model']}")
            print(f"   Type: {row['Type']}")
            print(f"   Accuracy: {row['Accuracy']:.4f}")
            print(f"   F1 Score: {row['F1 Score']:.4f}")
            print(f"   AUC: {row['AUC']:.4f}")
            print()
        
        return df_final
    
    def create_confusion_matrices(self):
        """Create confusion matrices for all models."""
        print("📊 Creating confusion matrices for all models...")
        
        if not self.test_data:
            print("❌ Test data not available")
            return
        
        y_test = self.test_data['y_test']
        
        # Calculate number of subplots needed
        n_models = len(self.all_results)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (model_name, results) in enumerate(self.all_results.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Get predictions
            y_pred = results.get('y_pred')
            if y_pred is None:
                ax.text(0.5, 0.5, 'No predictions\\navailable', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(model_name, fontweight='bold')
                continue
            
            # Create confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Plot heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
            
            # Calculate additional metrics
            accuracy = results.get('accuracy', accuracy_score(y_test, y_pred))
            f1 = results.get('f1_score', f1_score(y_test, y_pred, average='weighted'))
            
            ax.set_title(f'{model_name}\\nAcc: {accuracy:.3f}, F1: {f1:.3f}', 
                        fontweight='bold', fontsize=10)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.output_dir / 'all_confusion_matrices.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrices saved to {save_path}")
        plt.show()
    
    def analyze_error_patterns(self):
        """Analyze error patterns across models."""
        print("🔍 Analyzing error patterns...")
        
        if not self.test_data:
            print("❌ Test data not available")
            return
        
        X_test = self.test_data['X_test']
        y_test = self.test_data['y_test']
        
        # Collect all predictions
        all_predictions = {}
        error_analysis = {}
        
        for model_name, results in self.all_results.items():
            y_pred = results.get('y_pred')
            if y_pred is not None:
                all_predictions[model_name] = y_pred
                
                # Find errors
                errors = (y_test != y_pred)
                error_indices = np.where(errors)[0]
                
                error_analysis[model_name] = {
                    'total_errors': len(error_indices),
                    'error_rate': len(error_indices) / len(y_test),
                    'error_indices': error_indices,
                    'false_positives': np.sum((y_test == 0) & (y_pred == 1)),
                    'false_negatives': np.sum((y_test == 1) & (y_pred == 0))
                }
        
        # Create error analysis visualization
        self._plot_error_analysis(error_analysis)
        
        # Find common errors across models
        if len(all_predictions) > 1:
            self._analyze_common_errors(all_predictions, X_test, y_test)
        
        return error_analysis
    
    def _plot_error_analysis(self, error_analysis):
        """Plot error analysis results."""
        if not error_analysis:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        model_names = list(error_analysis.keys())
        error_rates = [error_analysis[name]['error_rate'] for name in model_names]
        false_positives = [error_analysis[name]['false_positives'] for name in model_names]
        false_negatives = [error_analysis[name]['false_negatives'] for name in model_names]
        
        # 1. Error rates
        bars = axes[0, 0].bar(range(len(model_names)), error_rates, alpha=0.7, color='red')
        axes[0, 0].set_title('Error Rates by Model', fontweight='bold')
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Error Rate')
        axes[0, 0].set_xticks(range(len(model_names)))
        axes[0, 0].set_xticklabels([name.replace(' ', '\\n') for name in model_names], rotation=45)
        
        # Add value labels
        for bar, rate in zip(bars, error_rates):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                          f'{rate:.3f}', ha='center', va='bottom')
        
        # 2. False positives vs false negatives
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, false_positives, width, label='False Positives', alpha=0.7)
        axes[0, 1].bar(x + width/2, false_negatives, width, label='False Negatives', alpha=0.7)
        axes[0, 1].set_title('False Positives vs False Negatives', fontweight='bold')
        axes[0, 1].set_xlabel('Models')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels([name.replace(' ', '\\n') for name in model_names], rotation=45)
        axes[0, 1].legend()
        
        # 3. Error distribution pie chart (for best model)
        best_model = min(error_analysis.keys(), key=lambda x: error_analysis[x]['error_rate'])
        best_errors = error_analysis[best_model]
        
        sizes = [best_errors['false_positives'], best_errors['false_negatives']]
        labels = ['False Positives\\n(Fake → Real)', 'False Negatives\\n(Real → Fake)']
        colors = ['lightcoral', 'lightskyblue']
        
        if sum(sizes) > 0:
            axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[1, 0].set_title(f'Error Distribution - {best_model}', fontweight='bold')
        else:
            axes[1, 0].text(0.5, 0.5, 'No errors detected!', ha='center', va='center', fontsize=14, fontweight='bold')
            axes[1, 0].set_title(f'Error Distribution - {best_model}', fontweight='bold')
        
        # 4. Model reliability comparison
        accuracy_scores = [1 - error_analysis[name]['error_rate'] for name in model_names]
        
        # Create a radar-like comparison
        angles = np.linspace(0, 2 * np.pi, len(model_names), endpoint=False)
        axes[1, 1].plot(angles, accuracy_scores, 'o-', linewidth=2, markersize=8)
        axes[1, 1].fill(angles, accuracy_scores, alpha=0.25)
        axes[1, 1].set_xticks(angles)
        axes[1, 1].set_xticklabels([name.replace(' ', '\\n') for name in model_names], fontsize=8)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_title('Model Reliability Comparison', fontweight='bold')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.output_dir / 'error_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"🔍 Error analysis saved to {save_path}")
        plt.show()
    
    def _analyze_common_errors(self, all_predictions, X_test, y_test):
        """Analyze common errors across models."""
        print("🔍 Analyzing common errors across models...")
        
        # Find samples that multiple models get wrong
        error_counts = np.zeros(len(y_test))
        
        for model_name, y_pred in all_predictions.items():
            errors = (y_test != y_pred)
            error_counts += errors.astype(int)
        
        # Find samples with most errors
        most_difficult = np.argsort(error_counts)[-10:]  # Top 10 most difficult samples
        
        print(f"\\n📋 Most Difficult Samples (misclassified by most models):")
        print("=" * 70)
        
        for i, idx in enumerate(most_difficult[::-1]):  # Reverse to show most difficult first
            # Use positional indexing to avoid index alignment issues
            count_val = error_counts.iloc[idx] if hasattr(error_counts, 'iloc') else error_counts[idx]
            if count_val > 0:
                print(f"{i+1}. Sample {idx}:")
                text_val = X_test.iloc[idx]
                print(f"   Text: {text_val[:100]}..." if isinstance(text_val, str) and len(text_val) > 100 else f"   Text: {text_val}")
                label_val = y_test.iloc[idx]
                print(f"   True label: {'Real' if label_val == 1 else 'Fake'}")
                print(f"   Misclassified by {int(count_val)}/{len(all_predictions)} models")
                print()
    
    def generate_final_report(self):
        """Generate a comprehensive final report."""
        print("📝 Generating comprehensive final report...")
        
        # Load all results
        if not self.all_results:
            self.load_all_results()
        
        # Create comparison DataFrame
        df_comparison = self.create_comprehensive_comparison()
        
        # Create confusion matrices
        self.create_confusion_matrices()
        
        # Analyze error patterns
        error_analysis = self.analyze_error_patterns()
        
        # Generate summary statistics
        summary = self._generate_summary_statistics(df_comparison)
        
        # Create final report document
        self._create_final_report_document(df_comparison, summary, error_analysis)
        
        print("✅ Comprehensive evaluation completed!")
        return df_comparison, summary, error_analysis
    
    def _generate_summary_statistics(self, df_comparison):
        """Generate summary statistics."""
        summary = {
            'total_models': len(df_comparison),
            'best_model': df_comparison.loc[df_comparison['F1 Score'].idxmax(), 'Model'],
            'best_f1_score': df_comparison['F1 Score'].max(),
            'average_accuracy': df_comparison['Accuracy'].mean(),
            'average_f1_score': df_comparison['F1 Score'].mean(),
            'models_above_95_f1': len(df_comparison[df_comparison['F1 Score'] > 0.95]),
            'bert_vs_traditional': {
                'bert_avg_f1': df_comparison[df_comparison['Type'] == 'BERT-based']['F1 Score'].mean(),
                'traditional_avg_f1': df_comparison[df_comparison['Type'] == 'Traditional ML']['F1 Score'].mean()
            }
        }
        return summary
    
    def _create_final_report_document(self, df_comparison, summary, error_analysis):
        """Create a comprehensive final report document."""
        report_content = f"""
# Kannada Fake News Detection - Comprehensive Model Evaluation Report

## Executive Summary

This report presents a comprehensive evaluation of {summary['total_models']} machine learning models 
trained for Kannada fake news detection. The models were evaluated on a balanced dataset with 
{len(self.test_data['y_test']) if self.test_data else 'N/A'} test samples.

### Key Findings:
- **Best Performing Model**: {summary['best_model']}
- **Best F1 Score**: {summary['best_f1_score']:.4f}
- **Average Model Accuracy**: {summary['average_accuracy']:.4f}
- **Models with >95% F1 Score**: {summary['models_above_95_f1']}/{summary['total_models']}

## Model Performance Comparison

### Top 3 Models:
{self._format_top_models(df_comparison)}

### Model Type Analysis:
- **BERT-based Models Average F1**: {summary['bert_vs_traditional']['bert_avg_f1']:.4f}
- **Traditional ML Average F1**: {summary['bert_vs_traditional']['traditional_avg_f1']:.4f}

## Error Analysis Summary

{self._format_error_analysis(error_analysis)}

## Technical Details

### Dataset Information:
- **Language**: Kannada
- **Task**: Binary classification (Fake vs Real news)
- **Training Samples**: {len(joblib.load('data/y_train.joblib')) if Path('data/y_train.joblib').exists() else 'N/A'}
- **Test Samples**: {len(self.test_data['y_test']) if self.test_data else 'N/A'}

### Model Categories:
1. **Traditional ML with TF-IDF**: Logistic Regression, Random Forest, SVM, Naive Bayes
2. **BERT-based**: Multilingual BERT with traditional ML classifiers

## Recommendations

1. **Production Deployment**: Use {summary['best_model']} for production deployment
2. **Resource Constraints**: For resource-constrained environments, traditional ML models perform excellently
3. **Ensemble Approach**: Consider ensemble methods combining top-performing models
4. **Continuous Monitoring**: Implement monitoring for model drift given the dynamic nature of fake news

## Files Generated:
- `comprehensive_model_comparison.png`: Visual comparison of all models
- `all_confusion_matrices.png`: Confusion matrices for all models  
- `error_analysis.png`: Error pattern analysis
- `model_comparison_table.csv`: Detailed metrics table
- `model_comparison_table.html`: Interactive HTML table

---
*Report generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # Save report
        report_path = self.output_dir / 'final_evaluation_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"📝 Final report saved to {report_path}")
    
    def _format_top_models(self, df_comparison):
        """Format top 3 models for the report."""
        top_3 = df_comparison.nlargest(3, 'F1 Score')
        formatted = ""
        for i, (_, row) in enumerate(top_3.iterrows(), 1):
            formatted += f"{i}. **{row['Model']}** - F1: {row['F1 Score']:.4f}, Accuracy: {row['Accuracy']:.4f}\\n"
        return formatted
    
    def _format_error_analysis(self, error_analysis):
        """Format error analysis for the report."""
        if not error_analysis:
            return "No error analysis available."
        
        best_model = min(error_analysis.keys(), key=lambda x: error_analysis[x]['error_rate'])
        best_errors = error_analysis[best_model]
        
        return f"""
**Best Model Error Analysis** ({best_model}):
- Total Errors: {best_errors['total_errors']}
- Error Rate: {best_errors['error_rate']:.4f}
- False Positives: {best_errors['false_positives']}
- False Negatives: {best_errors['false_negatives']}
"""


def main():
    """Main function to run comprehensive evaluation."""
    print("🚀 Starting Comprehensive Model Evaluation")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Generate complete evaluation
    df_comparison, summary, error_analysis = evaluator.generate_final_report()
    
    print("\\n" + "=" * 60)
    print("🎉 COMPREHENSIVE EVALUATION COMPLETED!")
    print("=" * 60)
    print(f"📊 Evaluated {summary['total_models']} models")
    print(f"🏆 Best Model: {summary['best_model']}")
    print(f"📈 Best F1 Score: {summary['best_f1_score']:.4f}")
    print(f"📁 All results saved to 'results/' directory")


if __name__ == "__main__":
    main()