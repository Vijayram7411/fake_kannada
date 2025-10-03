"""
Multilingual Fake News Classifier
Implements transformer-based binary classification for fake news detection
Supporting mBERT and XLM-RoBERTa as specified in the project report
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    BertTokenizer, BertForSequenceClassification,
    XLMRobertaTokenizer, XLMRobertaForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup
)
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultilingualNewsDataset(Dataset):
    """
    Custom Dataset class for multilingual news data
    """
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        """
        Initialize dataset
        
        Args:
            texts: List of news article texts
            labels: List of binary labels (0 for real, 1 for fake)
            tokenizer: Transformer tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class MultilingualFakeNewsClassifier(nn.Module):
    """
    Multilingual Fake News Classifier using transformer models
    """
    
    def __init__(self, model_name: str = 'bert-base-multilingual-cased', 
                 num_classes: int = 2, dropout_rate: float = 0.3):
        """
        Initialize the classifier
        
        Args:
            model_name: Name of the transformer model
            num_classes: Number of output classes (2 for binary classification)
            dropout_rate: Dropout rate for regularization
        """
        super(MultilingualFakeNewsClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load transformer model
        if 'xlm-roberta' in model_name.lower():
            self.transformer = XLMRobertaForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=num_classes,
                output_attentions=False,
                output_hidden_states=False
            )
        elif 'bert' in model_name.lower():
            self.transformer = BertForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                output_attentions=False,
                output_hidden_states=False
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Additional classification layers
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Classification logits
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        return outputs.logits

class MultilingualFakeNewsDetector:
    """
    Main class for multilingual fake news detection system
    """
    
    def __init__(self, model_name: str = 'bert-base-multilingual-cased',
                 max_length: int = 512, device: str = None):
        """
        Initialize the detector
        
        Args:
            model_name: Transformer model name
            max_length: Maximum sequence length
            device: Computing device ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer
        self._load_tokenizer()
        
        # Initialize model
        self.model = None
        self.training_history = []
        
    def _load_tokenizer(self):
        """Load the appropriate tokenizer"""
        if 'xlm-roberta' in self.model_name.lower():
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(self.model_name)
        elif 'bert' in self.model_name.lower():
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
    
    def prepare_data(self, texts: List[str], labels: List[int], 
                    test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Prepare data for training
        
        Args:
            texts: List of news texts
            labels: List of binary labels
            test_size: Proportion of test data
            random_state: Random seed
            
        Returns:
            Tuple of train and test datasets
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, 
            stratify=labels
        )
        
        # Create datasets
        train_dataset = MultilingualNewsDataset(
            X_train, y_train, self.tokenizer, self.max_length
        )
        test_dataset = MultilingualNewsDataset(
            X_test, y_test, self.tokenizer, self.max_length
        )
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")
        
        return train_dataset, test_dataset, X_test, y_test
    
    def train(self, train_dataset: MultilingualNewsDataset, 
             val_dataset: Optional[MultilingualNewsDataset] = None,
             batch_size: int = 16, epochs: int = 3, learning_rate: float = 2e-5,
             warmup_steps: int = 0, save_path: str = None) -> Dict:
        """
        Train the multilingual fake news classifier
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            batch_size: Batch size for training
            epochs: Number of training epochs
            learning_rate: Learning rate
            warmup_steps: Warmup steps for learning rate scheduler
            save_path: Path to save the trained model
            
        Returns:
            Training history dictionary
        """
        logger.info("Starting training...")
        
        # Initialize model
        self.model = MultilingualFakeNewsClassifier(
            model_name=self.model_name,
            num_classes=2
        ).to(self.device)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )
        
        # Set up optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_progress = tqdm(train_loader, desc="Training")
            for batch in train_progress:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # Update progress bar
                train_progress.set_postfix({
                    'loss': loss.item(),
                    'acc': train_correct / train_total
                })
            
            # Calculate epoch metrics
            epoch_train_loss = train_loss / len(train_loader)
            epoch_train_acc = train_correct / train_total
            
            training_history['train_loss'].append(epoch_train_loss)
            training_history['train_acc'].append(epoch_train_acc)
            
            # Validation phase
            if val_loader:
                val_loss, val_acc = self._validate(val_loader, criterion)
                training_history['val_loss'].append(val_loss)
                training_history['val_acc'].append(val_acc)
                
                logger.info(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
                logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                logger.info(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
        
        # Save model if path provided
        if save_path:
            self.save_model(save_path)
        
        self.training_history = training_history
        logger.info("Training completed!")
        
        return training_history
    
    def _validate(self, val_loader: DataLoader, criterion) -> Tuple[float, float]:
        """
        Validation phase
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Tuple of (validation_loss, validation_accuracy)
        """
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        return val_loss / len(val_loader), val_correct / val_total
    
    def predict(self, texts: List[str], batch_size: int = 16) -> Tuple[List[int], List[float]]:
        """
        Make predictions on new texts
        
        Args:
            texts: List of text strings
            batch_size: Batch size for inference
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Train first or load a saved model.")
        
        # Create dataset for prediction
        dummy_labels = [0] * len(texts)  # Dummy labels for dataset creation
        predict_dataset = MultilingualNewsDataset(
            texts, dummy_labels, self.tokenizer, self.max_length
        )
        
        predict_loader = DataLoader(
            predict_dataset, batch_size=batch_size, shuffle=False
        )
        
        self.model.eval()
        predictions = []
        confidence_scores = []
        
        with torch.no_grad():
            for batch in tqdm(predict_loader, desc="Predicting"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Get predictions and confidence scores
                _, predicted = torch.max(outputs.data, 1)
                max_probs, _ = torch.max(probabilities, 1)
                
                predictions.extend(predicted.cpu().numpy())
                confidence_scores.extend(max_probs.cpu().numpy())
        
        return predictions, confidence_scores
    
    def evaluate(self, test_texts: List[str], test_labels: List[int], 
                batch_size: int = 16) -> Dict:
        """
        Evaluate model performance
        
        Args:
            test_texts: Test text data
            test_labels: Test labels
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Evaluating model...")
        
        # Make predictions
        start_time = time.time()
        predictions, confidence_scores = self.predict(test_texts, batch_size)
        inference_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            test_labels, predictions, average='weighted'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            test_labels, predictions, average=None
        )
        
        # Confusion matrix
        cm = confusion_matrix(test_labels, predictions)
        
        # Performance metrics
        avg_inference_time = inference_time / len(test_texts)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'total_inference_time': inference_time,
            'avg_inference_time_per_sample': avg_inference_time,
            'samples_per_second': len(test_texts) / inference_time,
            'confidence_scores': confidence_scores
        }
        
        # Print results
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        logger.info(f"Average inference time: {avg_inference_time:.4f} seconds")
        logger.info(f"Samples per second: {len(test_texts) / inference_time:.2f}")
        
        # Check performance requirements from report
        performance_target = 2.0  # seconds
        accuracy_target = 0.90    # 90%
        
        if avg_inference_time <= performance_target:
            logger.info(f"✓ Performance requirement met: {avg_inference_time:.4f}s <= {performance_target}s")
        else:
            logger.warning(f"✗ Performance requirement not met: {avg_inference_time:.4f}s > {performance_target}s")
        
        if accuracy >= accuracy_target:
            logger.info(f"✓ Accuracy requirement met: {accuracy:.4f} >= {accuracy_target}")
        else:
            logger.warning(f"✗ Accuracy requirement not met: {accuracy:.4f} < {accuracy_target}")
        
        return results
    
    def cross_validate(self, texts: List[str], labels: List[int], 
                      k_folds: int = 5, **train_kwargs) -> Dict:
        """
        Perform k-fold cross-validation
        
        Args:
            texts: All text data
            labels: All labels
            k_folds: Number of folds
            **train_kwargs: Additional training arguments
            
        Returns:
            Cross-validation results
        """
        logger.info(f"Starting {k_folds}-fold cross-validation...")
        
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        cv_results = {
            'fold_accuracies': [],
            'fold_f1_scores': [],
            'fold_precisions': [],
            'fold_recalls': [],
            'mean_accuracy': 0,
            'mean_f1': 0,
            'mean_precision': 0,
            'mean_recall': 0,
            'std_accuracy': 0,
            'std_f1': 0
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
            logger.info(f"Fold {fold + 1}/{k_folds}")
            
            # Split data
            train_texts = [texts[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            val_texts = [texts[i] for i in val_idx]
            val_labels = [labels[i] for i in val_idx]
            
            # Create datasets
            train_dataset = MultilingualNewsDataset(
                train_texts, train_labels, self.tokenizer, self.max_length
            )
            val_dataset = MultilingualNewsDataset(
                val_texts, val_labels, self.tokenizer, self.max_length
            )
            
            # Train model
            self.train(train_dataset, val_dataset, **train_kwargs)
            
            # Evaluate
            results = self.evaluate(val_texts, val_labels)
            
            cv_results['fold_accuracies'].append(results['accuracy'])
            cv_results['fold_f1_scores'].append(results['f1_score'])
            cv_results['fold_precisions'].append(results['precision'])
            cv_results['fold_recalls'].append(results['recall'])
        
        # Calculate means and standard deviations
        cv_results['mean_accuracy'] = np.mean(cv_results['fold_accuracies'])
        cv_results['mean_f1'] = np.mean(cv_results['fold_f1_scores'])
        cv_results['mean_precision'] = np.mean(cv_results['fold_precisions'])
        cv_results['mean_recall'] = np.mean(cv_results['fold_recalls'])
        cv_results['std_accuracy'] = np.std(cv_results['fold_accuracies'])
        cv_results['std_f1'] = np.std(cv_results['fold_f1_scores'])
        
        logger.info(f"Cross-validation completed!")
        logger.info(f"Mean Accuracy: {cv_results['mean_accuracy']:.4f} (±{cv_results['std_accuracy']:.4f})")
        logger.info(f"Mean F1-Score: {cv_results['mean_f1']:.4f} (±{cv_results['std_f1']:.4f})")
        
        return cv_results
    
    def save_model(self, save_path: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'max_length': self.max_length,
            'training_history': self.training_history
        }, save_path)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, model_path: str):
        """Load a saved model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = MultilingualFakeNewsClassifier(
            model_name=checkpoint['model_name'],
            num_classes=2
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.model_name = checkpoint['model_name']
        self.max_length = checkpoint['max_length']
        self.training_history = checkpoint.get('training_history', [])
        
        logger.info(f"Model loaded from {model_path}")

# Example usage and testing
if __name__ == "__main__":
    # Sample multilingual data for testing
    sample_texts = [
        "ಇದು ನಿಜವಾದ ಸುದ್ದಿಯಾಗಿದೆ. ಸರ್ಕಾರವು ಹೊಸ ನೀತಿಯನ್ನು ಘೋಷಿಸಿದೆ.",  # Real Kannada
        "ಈ ಸುದ್ದಿ ಸಂಪೂರ್ಣವಾಗಿ ಸುಳ್ಳು! ಇದನ್ನು ನಂಬಬೇಡಿ!!",  # Fake Kannada
        "This is a legitimate news article about government policy changes.",  # Real English
        "BREAKING: Aliens have landed! Government hiding the truth!!!",  # Fake English
        "सरकार ने नई नीति की घोषणा की है। यह सत्य समाचार है।",  # Real Hindi
        "यह पूरी तरह से झूठी खबर है! इसे मत मानिए!!!"  # Fake Hindi
    ]
    
    sample_labels = [0, 1, 0, 1, 0, 1]  # 0 for real, 1 for fake
    
    print("Testing Multilingual Fake News Classifier:")
    print("=" * 50)
    
    # Initialize detector
    detector = MultilingualFakeNewsDetector(
        model_name='bert-base-multilingual-cased',
        max_length=256  # Shorter for demo
    )
    
    # Prepare data
    train_dataset, test_dataset, X_test, y_test = detector.prepare_data(
        sample_texts, sample_labels, test_size=0.3
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Train model (with very small settings for demo)
    training_history = detector.train(
        train_dataset,
        test_dataset,
        batch_size=2,
        epochs=1,
        learning_rate=2e-5
    )
    
    print("Training completed!")
    
    # Evaluate model
    results = detector.evaluate(X_test, y_test, batch_size=2)
    
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print(f"Test F1-Score: {results['f1_score']:.4f}")
    print(f"Average inference time: {results['avg_inference_time_per_sample']:.6f} seconds")