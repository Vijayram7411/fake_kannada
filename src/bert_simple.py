"""
Simplified BERT Feature Extraction for Kannada Fake News Detection
=================================================================

This module implements BERT feature extraction with traditional ML models
for Kannada fake news detection, focusing on efficiency and performance.
"""

import pandas as pd
import numpy as np
import joblib
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {device}")

class SimpleBERTExtractor:
    """
    Simple BERT feature extraction for Kannada text.
    """
    
def __init__(self, model_name='l3cube-pune/kannada-bert', max_length=128):
        """
        Initialize BERT feature extractor.
        
        Parameters:
        -----------
        model_name : str
            Name of the BERT model to use
        max_length : int
            Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length
        
        print(f"🤖 Initializing Simple BERT Extractor:")
        print(f"   - Model: {model_name}")
        print(f"   - Max length: {max_length}")
        print(f"   - Device: {device}")
        
        # Load tokenizer and model
        print("📥 Loading BERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
        print("✅ BERT model loaded successfully")
    
    def extract_embeddings(self, texts, batch_size=8):
        """
        Extract BERT embeddings for a list of texts.
        
        Parameters:
        -----------
        texts : list
            List of texts to process
        batch_size : int
            Batch size for processing
            
        Returns:
        --------
        np.array
            BERT embeddings (768-dimensional vectors)
        """
        print(f"🔄 Extracting BERT embeddings for {len(texts)} texts...")
        
        embeddings = []
        
        # Process in batches to avoid memory issues
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
            batch_texts = texts[i:i + batch_size]
            
            # Convert to strings and handle any None values
            batch_texts = [str(text) if text is not None else "" for text in batch_texts]
            
            try:
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                ).to(device)
                
                # Extract embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use [CLS] token embeddings (first token)
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    embeddings.extend(batch_embeddings)
                    
            except Exception as e:
                print(f"   ⚠️ Error processing batch {i//batch_size + 1}: {e}")
                # Add zero embeddings for failed batch
                zero_embeddings = np.zeros((len(batch_texts), 768))
                embeddings.extend(zero_embeddings)
        
        embeddings = np.array(embeddings)
        print(f"✅ Embedding extraction completed. Shape: {embeddings.shape}")
        
        return embeddings
    
    def save_embeddings(self, embeddings, path):
        """Save embeddings to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, embeddings)
        print(f"💾 Embeddings saved to {path}")
    
    def load_embeddings(self, path):
        """Load embeddings from disk."""
        embeddings = np.load(path)
        print(f"📂 Embeddings loaded from {path}. Shape: {embeddings.shape}")
        return embeddings


def train_bert_ml_models(X_train_embeddings, y_train, X_test_embeddings, y_test):
    """
    Train traditional ML models on BERT embeddings.
    
    Parameters:
    -----------
    X_train_embeddings, X_test_embeddings : np.array
        BERT embeddings for train/test
    y_train, y_test : arrays
        Labels for train/test
        
    Returns:
    --------
    dict
        Results from all models
    """
    print("🤖 Training ML models on BERT embeddings...")
    
    # Define models
    models = {
        'BERT + Logistic Regression': LogisticRegression(
            random_state=42, 
            max_iter=1000, 
            class_weight='balanced'
        ),
        'BERT + Random Forest': RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            class_weight='balanced'
        ),
        'BERT + SVM': SVC(
            kernel='rbf', 
            random_state=42, 
            class_weight='balanced',
            probability=True
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"   🎯 Training {name}...")
        
        try:
            # Train model
            model.fit(X_train_embeddings, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_embeddings)
            y_pred_proba = model.predict_proba(X_test_embeddings)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Store results
            results[name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'model': model,
                'classification_report': classification_report(y_test, y_pred)
            }
            
            print(f"   ✅ {name}: Accuracy={accuracy:.4f}, F1={f1:.4f}")
            
            # Save model
            model_path = f'models/bert_{name.lower().replace(" ", "_").replace("+", "")}.joblib'
            joblib.dump(model, model_path)
            
        except Exception as e:
            print(f"   ❌ {name} failed: {e}")
    
    return results


def plot_bert_results(results, y_test, save_dir='results'):
    """
    Create visualizations for BERT model results.
    
    Parameters:
    -----------
    results : dict
        Model results
    y_test : array
        True test labels
    save_dir : str
        Directory to save plots
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # 1. Model performance comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Performance metrics
    model_names = []
    accuracies = []
    f1_scores = []
    
    for name, metrics in results.items():
        model_names.append(name.replace('BERT + ', ''))
        accuracies.append(metrics['accuracy'])
        f1_scores.append(metrics['f1_score'])
    
    x = np.arange(len(model_names))
    width = 0.35
    
    axes[0].bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
    axes[0].bar(x + width/2, f1_scores, width, label='F1 Score', alpha=0.8)
    axes[0].set_xlabel('Models')
    axes[0].set_ylabel('Score')
    axes[0].set_title('BERT + Traditional ML Performance', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=45)
    axes[0].legend()
    axes[0].set_ylim(0, 1)
    
    # Add value labels on bars
    for i, (acc, f1) in enumerate(zip(accuracies, f1_scores)):
        axes[0].text(i - width/2, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
        axes[0].text(i + width/2, f1 + 0.01, f'{f1:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Confusion matrix for best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
    best_results = results[best_model_name]
    
    cm = confusion_matrix(y_test, best_results['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    axes[1].set_title(f'Confusion Matrix - {best_model_name}\\nAccuracy: {best_results["accuracy"]:.3f}', 
                     fontweight='bold')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(save_path / 'bert_results.png', dpi=300, bbox_inches='tight')
    print(f"📊 BERT results plot saved to {save_path / 'bert_results.png'}")
    plt.show()


def main():
    """Main function to run BERT + Traditional ML pipeline."""
    print("🚀 Starting BERT + Traditional ML Pipeline")
    print("=" * 60)
    
    # Load preprocessed data
    print("📂 Loading preprocessed data...")
    X_train = joblib.load('data/X_train.joblib')
    X_test = joblib.load('data/X_test.joblib')
    y_train = joblib.load('data/y_train.joblib')
    y_test = joblib.load('data/y_test.joblib')
    
    print(f"✅ Data loaded:")
    print(f"   - Training samples: {len(X_train)}")
    print(f"   - Test samples: {len(X_test)}")
    
    # Initialize BERT extractor
    print(f"\n{'='*60}")
    bert_extractor = SimpleBERTExtractor(
        model_name='bert-base-multilingual-cased',
        max_length=128
    )
    
    # Check if embeddings already exist
    train_embeddings_path = 'models/train_bert_embeddings.npy'
    test_embeddings_path = 'models/test_bert_embeddings.npy'
    
    if Path(train_embeddings_path).exists() and Path(test_embeddings_path).exists():
        print("\n📂 Loading existing BERT embeddings...")
        X_train_embeddings = bert_extractor.load_embeddings(train_embeddings_path)
        X_test_embeddings = bert_extractor.load_embeddings(test_embeddings_path)
    else:
        # Extract embeddings
        print(f"\n{'='*60}")
        print("🔄 Extracting BERT embeddings (this may take a few minutes)...")
        
        X_train_embeddings = bert_extractor.extract_embeddings(
            X_train.tolist(), batch_size=8
        )
        
        X_test_embeddings = bert_extractor.extract_embeddings(
            X_test.tolist(), batch_size=8
        )
        
        # Save embeddings for future use
        bert_extractor.save_embeddings(X_train_embeddings, train_embeddings_path)
        bert_extractor.save_embeddings(X_test_embeddings, test_embeddings_path)
    
    # Train traditional ML models on BERT embeddings
    print(f"\n{'='*60}")
    results = train_bert_ml_models(
        X_train_embeddings, y_train, 
        X_test_embeddings, y_test
    )
    
    # Print detailed results
    print(f"\n{'='*60}")
    print("📊 BERT + TRADITIONAL ML RESULTS")
    print("=" * 60)
    
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   F1 Score:  {metrics['f1_score']:.4f}")
        print(f"   Classification Report:")
        print("   " + metrics['classification_report'].replace('\n', '\n   '))
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   Best F1 Score: {results[best_model_name]['f1_score']:.4f}")
    
    # Create visualizations
    print(f"\n{'='*60}")
    print("📊 Creating visualizations...")
    plot_bert_results(results, y_test)
    
    # Save all results
    results_path = Path('models') / 'bert_results.joblib'
    joblib.dump(results, results_path)
    print(f"💾 All results saved to {results_path}")
    
    print(f"\n{'='*60}")
    print("🎉 BERT + TRADITIONAL ML PIPELINE COMPLETED!")
    print("=" * 60)
    print(f"✅ Best performing model: {best_model_name}")
    print(f"✅ Best F1 Score: {results[best_model_name]['f1_score']:.4f}")
    print(f"✅ All models and embeddings saved to 'models/' directory")
    print(f"✅ Visualizations saved to 'results/' directory")


if __name__ == "__main__":
    main()