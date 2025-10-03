"""
Baseline Models for Kannada Fake News Detection
==============================================

This module implements TF-IDF feature extraction and traditional ML models
as baseline approaches for Kannada fake news detection.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

class TFIDFFeatureExtractor:
    """
    TF-IDF feature extraction specifically designed for Kannada text.
    """
    
    def __init__(self, max_features=5000, ngram_range=(1, 2), 
                 min_df=2, max_df=0.95, sublinear_tf=True):
        """
        Initialize TF-IDF vectorizer with optimized parameters for Kannada.
        
        Parameters:
        -----------
        max_features : int
            Maximum number of features
        ngram_range : tuple
            Range of n-grams to extract
        min_df : int
            Minimum document frequency
        max_df : float
            Maximum document frequency (as ratio)
        sublinear_tf : bool
            Apply sublinear tf scaling
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            strip_accents=None,  # Don't strip accents for Kannada
            lowercase=False,     # Don't lowercase for Kannada
            token_pattern=r'\b\w+\b'  # Basic word tokenization
        )
        
        self.feature_names = None
        self.vocabulary_size = None
        
        print(f"🔧 Initialized TF-IDF Vectorizer:")
        print(f"   - Max features: {max_features}")
        print(f"   - N-gram range: {ngram_range}")
        print(f"   - Min document frequency: {min_df}")
        print(f"   - Max document frequency: {max_df}")
    
    def fit_transform(self, X_train):
        """
        Fit vectorizer on training data and transform.
        
        Parameters:
        -----------
        X_train : array-like
            Training texts
            
        Returns:
        --------
        scipy.sparse matrix
            TF-IDF features for training data
        """
        print("🔄 Fitting TF-IDF vectorizer on training data...")
        
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.vocabulary_size = len(self.feature_names)
        
        print(f"✅ TF-IDF vectorization completed:")
        print(f"   - Vocabulary size: {self.vocabulary_size}")
        print(f"   - Training matrix shape: {X_train_tfidf.shape}")
        print(f"   - Sparsity: {1 - X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1]):.3f}")
        
        return X_train_tfidf
    
    def transform(self, X_test):
        """
        Transform test data using fitted vectorizer.
        
        Parameters:
        -----------
        X_test : array-like
            Test texts
            
        Returns:
        --------
        scipy.sparse matrix
            TF-IDF features for test data
        """
        print("🔄 Transforming test data...")
        
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        print(f"✅ Test data transformation completed:")
        print(f"   - Test matrix shape: {X_test_tfidf.shape}")
        
        return X_test_tfidf
    
    def get_top_features(self, n_features=20):
        """
        Get top features by TF-IDF score.
        
        Parameters:
        -----------
        n_features : int
            Number of top features to return
            
        Returns:
        --------
        list
            Top feature names
        """
        if self.feature_names is None:
            return []
        
        # Get feature indices sorted by their importance
        # This is a simple approach - in practice you might want to use
        # feature importance from trained models
        return list(self.feature_names[:n_features])
    
    def save_vectorizer(self, path):
        """Save the fitted vectorizer."""
        joblib.dump(self.vectorizer, path)
        print(f"💾 Vectorizer saved to {path}")
    
    def load_vectorizer(self, path):
        """Load a fitted vectorizer."""
        self.vectorizer = joblib.load(path)
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.vocabulary_size = len(self.feature_names)
        print(f"📂 Vectorizer loaded from {path}")


class BaselineModels:
    """
    Collection of baseline ML models for fake news detection.
    """
    
    def __init__(self):
        """Initialize baseline models with optimized hyperparameters."""
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=42, 
                max_iter=1000, 
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                class_weight='balanced'
            ),
            'SVM': SVC(
                kernel='linear', 
                random_state=42, 
                class_weight='balanced', 
                probability=True
            ),
            'Naive Bayes': MultinomialNB(alpha=0.1)
        }
        
        self.trained_models = {}
        self.model_results = {}
        
        print(f"🤖 Initialized {len(self.models)} baseline models:")
        for name in self.models.keys():
            print(f"   - {name}")
    
    def train_models(self, X_train, y_train):
        """
        Train all baseline models.
        
        Parameters:
        -----------
        X_train : sparse matrix
            Training features
        y_train : array-like
            Training labels
        """
        print("🎯 Training baseline models...")
        
        for name, model in self.models.items():
            print(f"   Training {name}...")
            
            try:
                model.fit(X_train, y_train)
                self.trained_models[name] = model
                print(f"   ✅ {name} training completed")
            except Exception as e:
                print(f"   ❌ {name} training failed: {e}")
        
        print(f"✅ Model training completed. {len(self.trained_models)} models trained successfully.")
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models.
        
        Parameters:
        -----------
        X_test : sparse matrix
            Test features
        y_test : array-like
            Test labels
            
        Returns:
        --------
        dict
            Evaluation results for each model
        """
        print("📊 Evaluating models...")
        
        results = {}
        
        for name, model in self.trained_models.items():
            print(f"   Evaluating {name}...")
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            auc = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'classification_report': classification_report(y_test, y_pred)
            }
            
            print(f"   ✅ {name}: Acc={accuracy:.3f}, F1={f1:.3f}, AUC={auc:.3f}")
        
        self.model_results = results
        return results
    
    def cross_validate_models(self, X_train, y_train, cv=5):
        """
        Perform cross-validation for all models.
        
        Parameters:
        -----------
        X_train : sparse matrix
            Training features
        y_train : array-like
            Training labels
        cv : int
            Number of cross-validation folds
            
        Returns:
        --------
        dict
            Cross-validation results
        """
        print(f"🔄 Performing {cv}-fold cross-validation...")
        
        cv_results = {}
        
        for name, model in self.trained_models.items():
            print(f"   Cross-validating {name}...")
            
            # Perform cross-validation
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted')
            
            cv_results[name] = {
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'scores': scores
            }
            
            print(f"   ✅ {name}: {scores.mean():.3f} ± {scores.std():.3f}")
        
        return cv_results
    
    def get_best_model(self):
        """Get the best performing model based on F1 score."""
        if not self.model_results:
            return None, None
        
        best_model_name = max(self.model_results.keys(), 
                             key=lambda x: self.model_results[x]['f1_score'])
        best_model = self.trained_models[best_model_name]
        
        print(f"🏆 Best model: {best_model_name} (F1: {self.model_results[best_model_name]['f1_score']:.3f})")
        
        return best_model_name, best_model
    
    def save_models(self, output_dir='models'):
        """Save all trained models."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for name, model in self.trained_models.items():
            model_path = output_path / f"{name.lower().replace(' ', '_')}.joblib"
            joblib.dump(model, model_path)
        
        # Save results
        results_path = output_path / 'baseline_results.joblib'
        joblib.dump(self.model_results, results_path)
        
        print(f"💾 Models and results saved to {output_path}")


def plot_model_comparison(results, save_path='results/model_comparison.png'):
    """
    Plot comparison of model performances.
    
    Parameters:
    -----------
    results : dict
        Model evaluation results
    save_path : str
        Path to save the plot
    """
    # Create comparison dataframe
    comparison_data = []
    for model_name, metrics in results.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1 Score': metrics['f1_score'],
            'AUC': metrics['auc']
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    
    for i, metric in enumerate(metrics):
        row = i // 3
        col = i % 3
        
        ax = axes[row, col]
        bars = ax.bar(df_comparison['Model'], df_comparison[metric], 
                     color=plt.cm.Set3(np.arange(len(df_comparison))))
        
        ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
        
        # Rotate x-axis labels
        ax.tick_params(axis='x', rotation=45)
    
    # Remove empty subplot
    axes[1, 2].remove()
    
    plt.tight_layout()
    
    # Save plot
    Path(save_path).parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Model comparison plot saved to {save_path}")
    plt.show()


def plot_confusion_matrices(results, y_test, save_path='results/confusion_matrices.png'):
    """
    Plot confusion matrices for all models.
    
    Parameters:
    -----------
    results : dict
        Model evaluation results
    y_test : array-like
        True test labels
    save_path : str
        Path to save the plot
    """
    n_models = len(results)
    cols = 2
    rows = (n_models + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 5))
    if n_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, (model_name, metrics) in enumerate(results.items()):
        ax = axes[i] if n_models > 1 else axes[0]
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, metrics['y_pred'])
        
        # Plot
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
        
        ax.set_title(f'{model_name}\nAccuracy: {metrics["accuracy"]:.3f}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    
    # Save plot
    Path(save_path).parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Confusion matrices saved to {save_path}")
    plt.show()


def main():
    """Main function to run the baseline model pipeline."""
    print("🚀 Starting Baseline Model Training Pipeline")
    print("=" * 50)
    
    # Load preprocessed data
    print("📂 Loading preprocessed data...")
    X_train = joblib.load('data/X_train.joblib')
    X_test = joblib.load('data/X_test.joblib')
    y_train = joblib.load('data/y_train.joblib')
    y_test = joblib.load('data/y_test.joblib')
    
    print(f"✅ Data loaded:")
    print(f"   - Training samples: {len(X_train)}")
    print(f"   - Test samples: {len(X_test)}")
    
    # Initialize TF-IDF vectorizer
    print("\n" + "=" * 50)
    tfidf_extractor = TFIDFFeatureExtractor(
        max_features=3000,  # Reduced for smaller dataset
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    # Extract features
    X_train_tfidf = tfidf_extractor.fit_transform(X_train)
    X_test_tfidf = tfidf_extractor.transform(X_test)
    
    # Save vectorizer
    tfidf_extractor.save_vectorizer('models/tfidf_vectorizer.joblib')
    
    # Initialize and train baseline models
    print("\n" + "=" * 50)
    baseline_models = BaselineModels()
    
    # Train models
    baseline_models.train_models(X_train_tfidf, y_train)
    
    # Evaluate models
    print("\n" + "=" * 50)
    results = baseline_models.evaluate_models(X_test_tfidf, y_test)
    
    # Cross-validation
    print("\n" + "=" * 50)
    cv_results = baseline_models.cross_validate_models(X_train_tfidf, y_train)
    
    # Print detailed results
    print("\n" + "=" * 50)
    print("📊 DETAILED RESULTS")
    print("=" * 50)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1 Score:  {metrics['f1_score']:.4f}")
        print(f"   AUC:       {metrics['auc']:.4f}")
        print(f"   CV Score:  {cv_results[model_name]['mean_score']:.4f} ± {cv_results[model_name]['std_score']:.4f}")
    
    # Get best model
    print("\n" + "=" * 50)
    best_model_name, best_model = baseline_models.get_best_model()
    
    # Save models
    baseline_models.save_models()
    
    # Create visualizations
    print("\n" + "=" * 50)
    print("📊 Creating visualizations...")
    plot_model_comparison(results)
    plot_confusion_matrices(results, y_test)
    
    # Print summary
    print("\n" + "=" * 50)
    print("🎉 BASELINE MODEL PIPELINE COMPLETED!")
    print("=" * 50)
    print(f"✅ Best performing model: {best_model_name}")
    print(f"✅ Best F1 Score: {results[best_model_name]['f1_score']:.4f}")
    print(f"✅ Models and results saved to 'models/' directory")
    print(f"✅ Visualizations saved to 'results/' directory")
    

if __name__ == "__main__":
    main()