"""
BERT-based Models for Kannada Fake News Detection
================================================

This module implements BERT-based approaches for Kannada fake news detection,
including both feature extraction and fine-tuning approaches.
"""

import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {device}")

class KannadaFakeNewsDataset(Dataset):
    """
    Dataset class for Kannada fake news detection with BERT tokenization.
    """
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        Initialize dataset.
        
        Parameters:
        -----------
        texts : list
            List of text samples
        labels : list
            List of labels (0=fake, 1=real)
        tokenizer : transformers tokenizer
            BERT tokenizer
        max_length : int
            Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BERTFeatureExtractor:
    """
    Extract BERT embeddings for use with traditional ML models.
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
        
        print(f"🤖 Initializing BERT Feature Extractor:")
        print(f"   - Model: {model_name}")
        print(f"   - Max length: {max_length}")
        print(f"   - Device: {device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
        print("✅ BERT model loaded successfully")
    
    def extract_embeddings(self, texts, batch_size=16):
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
            BERT embeddings
        """
        print(f"🔄 Extracting BERT embeddings for {len(texts)} texts...")
        
        embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
            batch_texts = texts[i:i + batch_size]
            
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
                # Use [CLS] token embeddings
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(batch_embeddings)
        
        embeddings = np.array(embeddings)
        print(f"✅ Embedding extraction completed. Shape: {embeddings.shape}")
        
        return embeddings
    
    def save_embeddings(self, embeddings, path):
        """Save embeddings to disk."""
        np.save(path, embeddings)
        print(f"💾 Embeddings saved to {path}")
    
    def load_embeddings(self, path):
        """Load embeddings from disk."""
        embeddings = np.load(path)
        print(f"📂 Embeddings loaded from {path}. Shape: {embeddings.shape}")
        return embeddings


class BERTClassifier(nn.Module):
    """
    BERT-based classifier for fake news detection.
    """
    
def __init__(self, model_name='l3cube-pune/kannada-bert', n_classes=2, dropout=0.3):
        """
        Initialize BERT classifier.
        
        Parameters:
        -----------
        model_name : str
            Name of the BERT model
        n_classes : int
            Number of output classes
        dropout : float
            Dropout rate
        """
        super(BERTClassifier, self).__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        """Forward pass."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)


class BERTTrainer:
    """
    Trainer class for BERT-based fake news detection.
    """
    
def __init__(self, model_name='l3cube-pune/kannada-bert', max_length=128):
        """
        Initialize BERT trainer.
        
        Parameters:
        -----------
        model_name : str
            Name of the BERT model
        max_length : int
            Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length
        
        print(f"🎯 Initializing BERT Trainer:")
        print(f"   - Model: {model_name}")
        print(f"   - Max length: {max_length}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        ).to(device)
        
        print("✅ BERT model initialized for fine-tuning")
    
    def prepare_datasets(self, X_train, y_train, X_test, y_test):
        """
        Prepare datasets for training.
        
        Parameters:
        -----------
        X_train, y_train : arrays
            Training data
        X_test, y_test : arrays
            Test data
            
        Returns:
        --------
        tuple
            Train and test datasets
        """
        print("🔄 Preparing datasets...")
        
        train_dataset = KannadaFakeNewsDataset(
            texts=X_train.tolist(),
            labels=y_train.tolist(),
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        test_dataset = KannadaFakeNewsDataset(
            texts=X_test.tolist(),
            labels=y_test.tolist(),
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        print(f"✅ Datasets prepared:")
        print(f"   - Train size: {len(train_dataset)}")
        print(f"   - Test size: {len(test_dataset)}")
        
        return train_dataset, test_dataset
    
    def train_model(self, train_dataset, eval_dataset, output_dir='models/bert_finetuned'):
        """
        Fine-tune BERT model.
        
        Parameters:
        -----------
        train_dataset : Dataset
            Training dataset
        eval_dataset : Dataset
            Evaluation dataset
        output_dir : str
            Directory to save the model
        """
        print("🎯 Starting BERT fine-tuning...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            save_total_limit=2,
            seed=42
        )
        
        # Custom compute metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return {
                'accuracy': accuracy_score(labels, predictions),
                'f1': f1_score(labels, predictions, average='weighted')
            }
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Train
        print("🔄 Training in progress...")
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"✅ BERT fine-tuning completed. Model saved to {output_dir}")
        
        return trainer
    
    def evaluate_model(self, trainer, test_dataset):
        """
        Evaluate the fine-tuned model.
        
        Parameters:
        -----------
        trainer : Trainer
            Trained model
        test_dataset : Dataset
            Test dataset
            
        Returns:
        --------
        dict
            Evaluation results
        """
        print("📊 Evaluating BERT model...")
        
        # Get predictions
        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'y_pred': y_pred,
            'y_true': y_true,
            'classification_report': classification_report(y_true, y_pred)
        }
        
        print(f"✅ BERT Evaluation Results:")
        print(f"   - Accuracy: {accuracy:.4f}")
        print(f"   - F1 Score: {f1:.4f}")
        
        return results


def bert_with_traditional_ml(X_train, y_train, X_test, y_test, 
                           model_name='bert-base-multilingual-cased'):
    """
    Use BERT embeddings with traditional ML models.
    
    Parameters:
    -----------
    X_train, y_train : arrays
        Training data
    X_test, y_test : arrays
        Test data
    model_name : str
        BERT model name
        
    Returns:
    --------
    dict
        Results from BERT + traditional ML
    """
    print("🔄 BERT + Traditional ML Approach")
    print("=" * 50)
    
    # Initialize BERT feature extractor
    bert_extractor = BERTFeatureExtractor(model_name=model_name, max_length=128)
    
    # Extract embeddings
    print("🔄 Extracting training embeddings...")
    X_train_embeddings = bert_extractor.extract_embeddings(X_train.tolist(), batch_size=8)
    
    print("🔄 Extracting test embeddings...")
    X_test_embeddings = bert_extractor.extract_embeddings(X_test.tolist(), batch_size=8)
    
    # Save embeddings
    Path('models').mkdir(exist_ok=True)
    bert_extractor.save_embeddings(X_train_embeddings, 'models/train_bert_embeddings.npy')
    bert_extractor.save_embeddings(X_test_embeddings, 'models/test_bert_embeddings.npy')
    
    # Train traditional ML models on BERT embeddings
    print("🤖 Training models on BERT embeddings...")
    
    models = {
        'BERT + Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'BERT + Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"   Training {name}...")
        
        # Train
        model.fit(X_train_embeddings, y_train)
        
        # Predict
        y_pred = model.predict(X_test_embeddings)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results[name] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'y_pred': y_pred,
            'model': model
        }
        
        print(f"   ✅ {name}: Acc={accuracy:.3f}, F1={f1:.3f}")
        
        # Save model
        joblib.dump(model, f'models/{name.lower().replace(" ", "_").replace("+", "")}.joblib')
    
    return results


def main():
    """Main function to run BERT-based approaches."""
    print("🚀 Starting BERT-based Model Training")
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
    
    # Check if we have enough data for BERT training
    if len(X_train) < 100:
        print("⚠️  Warning: Small dataset detected. BERT fine-tuning might not be optimal.")
        print("   Proceeding with BERT + Traditional ML approach only.")
        
        # BERT + Traditional ML approach
        bert_traditional_results = bert_with_traditional_ml(
            X_train, y_train, X_test, y_test
        )
        
        print("\n" + "=" * 50)
        print("📊 BERT + Traditional ML Results:")
        print("=" * 50)
        
        for name, metrics in bert_traditional_results.items():
            print(f"{name}:")
            print(f"   Accuracy: {metrics['accuracy']:.4f}")
            print(f"   F1 Score: {metrics['f1_score']:.4f}")
        
    else:
        # Full BERT fine-tuning approach
        print("\n" + "=" * 50)
        print("🎯 BERT Fine-tuning Approach")
        print("=" * 50)
        
        # Initialize BERT trainer
        bert_trainer = BERTTrainer(
            model_name='l3cube-pune/kannada-bert',
            max_length=128
        )
        
        # Prepare datasets
        train_dataset, test_dataset = bert_trainer.prepare_datasets(
            X_train, y_train, X_test, y_test
        )
        
        # Train model
        trainer = bert_trainer.train_model(train_dataset, test_dataset)
        
        # Evaluate
        bert_results = bert_trainer.evaluate_model(trainer, test_dataset)
        
        print("\n" + "=" * 50)
        print("📊 BERT Fine-tuning Results:")
        print("=" * 50)
        print(f"Accuracy: {bert_results['accuracy']:.4f}")
        print(f"F1 Score: {bert_results['f1_score']:.4f}")
        
        # Also run BERT + Traditional ML for comparison
        print("\n" + "=" * 50)
        bert_traditional_results = bert_with_traditional_ml(
            X_train, y_train, X_test, y_test
        )
        
        # Compare all approaches
        print("\n" + "=" * 50)
        print("📊 COMPARISON OF ALL BERT APPROACHES:")
        print("=" * 50)
        
        print(f"BERT Fine-tuning:")
        print(f"   Accuracy: {bert_results['accuracy']:.4f}")
        print(f"   F1 Score: {bert_results['f1_score']:.4f}")
        
        for name, metrics in bert_traditional_results.items():
            print(f"{name}:")
            print(f"   Accuracy: {metrics['accuracy']:.4f}")
            print(f"   F1 Score: {metrics['f1_score']:.4f}")
    
    print("\n" + "=" * 50)
    print("🎉 BERT-BASED APPROACHES COMPLETED!")
    print("=" * 50)


if __name__ == "__main__":
    main()