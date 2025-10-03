"""
Multilingual Feature Extractor for Fake News Detection
Uses mBERT and XLM-RoBERTa as specified in the project report
"""

import torch
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer, AutoModel,
    BertTokenizer, BertModel,
    XLMRobertaTokenizer, XLMRobertaModel
)
from typing import List, Dict, Tuple, Union, Optional
import logging
from tqdm import tqdm
import gc
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultilingualFeatureExtractor:
    """
    Advanced feature extractor using transformer models for multilingual fake news detection
    Supports mBERT and XLM-RoBERTa models as specified in the project methodology
    """
    
    def __init__(self, model_name: str = 'bert-base-multilingual-cased', 
                 max_length: int = 512, device: str = None):
        """
        Initialize the feature extractor
        
        Args:
            model_name: Name of the transformer model to use
            max_length: Maximum sequence length for tokenization
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Loading model: {model_name}")
        
        # Load tokenizer and model based on model name
        self._load_model()
        
        # Model configuration
        self.embedding_dim = self.model.config.hidden_size
        
        logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
    
    def _load_model(self):
        """Load the specified transformer model"""
        try:
            if 'xlm-roberta' in self.model_name.lower():
                self.tokenizer = XLMRobertaTokenizer.from_pretrained(self.model_name)
                self.model = XLMRobertaModel.from_pretrained(self.model_name)
            elif 'bert' in self.model_name.lower():
                self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
                self.model = BertModel.from_pretrained(self.model_name)
            else:
                # Generic approach for other models
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {str(e)}")
            raise
    
    def tokenize_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenize a batch of texts for the transformer model
        
        Args:
            texts: List of text strings to tokenize
            
        Returns:
            Dictionary containing tokenized inputs
        """
        # Handle empty or None texts
        cleaned_texts = []
        for text in texts:
            if text is None or text == '':
                cleaned_texts.append('[UNK]')
            else:
                cleaned_texts.append(str(text))
        
        # Tokenize
        encoded = self.tokenizer(
            cleaned_texts,
            add_special_tokens=True,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True
        )
        
        # Move to device
        for key in encoded:
            encoded[key] = encoded[key].to(self.device)
        
        return encoded
    
    def extract_embeddings(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Extract transformer embeddings for a list of texts
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            
        Returns:
            numpy array of embeddings
        """
        all_embeddings = []
        
        # Process in batches to manage memory
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            encoded = self.tokenize_texts(batch_texts)
            
            # Extract embeddings
            with torch.no_grad():
                outputs = self.model(**encoded)
                
                # Use [CLS] token embedding (first token) for classification
                # or mean pooling of all tokens
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    # Use pooler output if available (BERT-style)
                    batch_embeddings = outputs.pooler_output
                else:
                    # Use mean pooling of last hidden states
                    last_hidden_state = outputs.last_hidden_state
                    attention_mask = encoded['attention_mask']
                    
                    # Mean pooling with attention mask
                    masked_embeddings = last_hidden_state * attention_mask.unsqueeze(-1)
                    batch_embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)
                
                # Convert to CPU and add to list
                all_embeddings.append(batch_embeddings.cpu().numpy())
            
            # Clear GPU memory
            del encoded, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate all embeddings
        embeddings = np.vstack(all_embeddings)
        
        logger.info(f"Extracted embeddings shape: {embeddings.shape}")
        return embeddings
    
    def extract_linguistic_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract linguistic features from texts
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of linguistic features
        """
        features = []
        
        for text in texts:
            text_features = self._compute_linguistic_features(str(text))
            features.append(list(text_features.values()))
        
        return np.array(features)
    
    def _compute_linguistic_features(self, text: str) -> Dict[str, float]:
        """
        Compute linguistic features for a single text
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary of linguistic features
        """
        import re
        import string
        from collections import Counter
        
        features = {}
        
        if not text or len(text.strip()) == 0:
            # Return zero features for empty text
            return {
                'text_length': 0, 'word_count': 0, 'sentence_count': 0,
                'avg_word_length': 0, 'punctuation_ratio': 0, 'uppercase_ratio': 0,
                'digit_ratio': 0, 'special_char_ratio': 0, 'exclamation_count': 0,
                'question_count': 0, 'unique_word_ratio': 0, 'repeated_char_ratio': 0,
                'kannada_char_ratio': 0, 'devanagari_char_ratio': 0, 'english_char_ratio': 0
            }
        
        # Basic text statistics
        features['text_length'] = len(text)
        words = text.split()
        features['word_count'] = len(words)
        features['sentence_count'] = len(re.split(r'[.!?]+', text))
        
        # Word-level features
        if words:
            features['avg_word_length'] = np.mean([len(word) for word in words])
            word_counts = Counter(words)
            features['unique_word_ratio'] = len(word_counts) / len(words)
        else:
            features['avg_word_length'] = 0
            features['unique_word_ratio'] = 0
        
        # Character-level features
        features['punctuation_ratio'] = sum(1 for c in text if c in string.punctuation) / len(text)
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text)
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text)
        features['special_char_ratio'] = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        
        # Specific punctuation counts
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        
        # Repeated character patterns (potential spam indicator)
        repeated_chars = len(re.findall(r'(.)\1{2,}', text))
        features['repeated_char_ratio'] = repeated_chars / len(text) if text else 0
        
        # Language-specific character ratios
        kannada_chars = len(re.findall(r'[\u0C80-\u0CFF]', text))
        devanagari_chars = len(re.findall(r'[\u0900-\u097F]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = len(re.sub(r'[^\w]', '', text))
        
        if total_chars > 0:
            features['kannada_char_ratio'] = kannada_chars / total_chars
            features['devanagari_char_ratio'] = devanagari_chars / total_chars
            features['english_char_ratio'] = english_chars / total_chars
        else:
            features['kannada_char_ratio'] = 0
            features['devanagari_char_ratio'] = 0
            features['english_char_ratio'] = 0
        
        return features
    
    def extract_stylometric_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract stylometric features as mentioned in the system design
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of stylometric features
        """
        features = []
        
        for text in texts:
            text_features = self._compute_stylometric_features(str(text))
            features.append(list(text_features.values()))
        
        return np.array(features)
    
    def _compute_stylometric_features(self, text: str) -> Dict[str, float]:
        """
        Compute stylometric features for author profiling
        """
        import re
        from textstat import flesch_reading_ease, flesch_kincaid_grade
        
        features = {}
        
        if not text or len(text.strip()) == 0:
            return {
                'avg_sentence_length': 0, 'sentence_length_variance': 0,
                'function_word_ratio': 0, 'content_word_ratio': 0,
                'type_token_ratio': 0, 'hapax_legomena_ratio': 0,
                'readability_score': 0, 'complexity_score': 0
            }
        
        # Sentence-level features
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if sentences:
            sentence_lengths = [len(s.split()) for s in sentences]
            features['avg_sentence_length'] = np.mean(sentence_lengths)
            features['sentence_length_variance'] = np.var(sentence_lengths)
        else:
            features['avg_sentence_length'] = 0
            features['sentence_length_variance'] = 0
        
        # Word-level features
        words = text.lower().split()
        if words:
            # Function words (common words that carry grammatical meaning)
            function_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should'}
            function_word_count = sum(1 for word in words if word in function_words)
            features['function_word_ratio'] = function_word_count / len(words)
            features['content_word_ratio'] = 1 - features['function_word_ratio']
            
            # Lexical diversity
            unique_words = set(words)
            features['type_token_ratio'] = len(unique_words) / len(words)
            
            # Hapax legomena (words that appear only once)
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            hapax_count = sum(1 for freq in word_freq.values() if freq == 1)
            features['hapax_legomena_ratio'] = hapax_count / len(words)
        else:
            features['function_word_ratio'] = 0
            features['content_word_ratio'] = 0
            features['type_token_ratio'] = 0
            features['hapax_legomena_ratio'] = 0
        
        # Readability features (works best for English)
        try:
            features['readability_score'] = flesch_reading_ease(text)
            features['complexity_score'] = flesch_kincaid_grade(text)
        except:
            features['readability_score'] = 50  # Default neutral score
            features['complexity_score'] = 10   # Default grade level
        
        return features
    
    def extract_psychological_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract psychological features as mentioned in the system design
        """
        features = []
        
        for text in texts:
            text_features = self._compute_psychological_features(str(text))
            features.append(list(text_features.values()))
        
        return np.array(features)
    
    def _compute_psychological_features(self, text: str) -> Dict[str, float]:
        """
        Compute psychological and emotional indicators
        """
        import re
        
        features = {}
        
        if not text or len(text.strip()) == 0:
            return {
                'emotion_word_ratio': 0, 'positive_word_ratio': 0, 'negative_word_ratio': 0,
                'urgency_indicator_ratio': 0, 'certainty_indicator_ratio': 0,
                'first_person_ratio': 0, 'second_person_ratio': 0, 'third_person_ratio': 0
            }
        
        words = text.lower().split()
        total_words = len(words) if words else 1
        
        # Emotional words (basic set)
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'happy', 'joy', 'success', 'win', 'victory', 'best', 'perfect'}
        negative_words = {'bad', 'terrible', 'awful', 'horrible', 'hate', 'angry', 'sad', 'fear', 'worried', 'concerned', 'problem', 'issue', 'crisis', 'disaster', 'fail', 'failure', 'worst', 'dangerous'}
        emotion_words = positive_words.union(negative_words)
        
        # Urgency indicators
        urgency_words = {'urgent', 'immediately', 'now', 'quickly', 'asap', 'emergency', 'breaking', 'alert', 'warning', 'must', 'should', 'need', 'important', 'critical'}
        
        # Certainty indicators
        certainty_words = {'always', 'never', 'definitely', 'certainly', 'absolutely', 'completely', 'totally', 'entirely', 'obviously', 'clearly', 'undoubtedly', 'surely'}
        
        # Personal pronouns
        first_person = {'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
        second_person = {'you', 'your', 'yours'}
        third_person = {'he', 'she', 'it', 'they', 'them', 'his', 'her', 'its', 'their', 'theirs'}
        
        # Calculate ratios
        emotion_count = sum(1 for word in words if word in emotion_words)
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        urgency_count = sum(1 for word in words if word in urgency_words)
        certainty_count = sum(1 for word in words if word in certainty_words)
        first_person_count = sum(1 for word in words if word in first_person)
        second_person_count = sum(1 for word in words if word in second_person)
        third_person_count = sum(1 for word in words if word in third_person)
        
        features['emotion_word_ratio'] = emotion_count / total_words
        features['positive_word_ratio'] = positive_count / total_words
        features['negative_word_ratio'] = negative_count / total_words
        features['urgency_indicator_ratio'] = urgency_count / total_words
        features['certainty_indicator_ratio'] = certainty_count / total_words
        features['first_person_ratio'] = first_person_count / total_words
        features['second_person_ratio'] = second_person_count / total_words
        features['third_person_ratio'] = third_person_count / total_words
        
        return features
    
    def extract_all_features(self, texts: List[str], include_embeddings: bool = True,
                           batch_size: int = 16) -> Tuple[np.ndarray, List[str]]:
        """
        Extract all types of features as specified in the methodology
        
        Args:
            texts: List of text strings
            include_embeddings: Whether to include transformer embeddings
            batch_size: Batch size for embedding extraction
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        logger.info("Extracting all features...")
        
        all_features = []
        feature_names = []
        
        # 1. Extract transformer embeddings
        if include_embeddings:
            logger.info("Extracting transformer embeddings...")
            embeddings = self.extract_embeddings(texts, batch_size)
            all_features.append(embeddings)
            feature_names.extend([f'embedding_{i}' for i in range(embeddings.shape[1])])
        
        # 2. Extract linguistic features
        logger.info("Extracting linguistic features...")
        linguistic_features = self.extract_linguistic_features(texts)
        all_features.append(linguistic_features)
        linguistic_feature_names = [
            'text_length', 'word_count', 'sentence_count', 'avg_word_length',
            'punctuation_ratio', 'uppercase_ratio', 'digit_ratio', 'special_char_ratio',
            'exclamation_count', 'question_count', 'unique_word_ratio', 'repeated_char_ratio',
            'kannada_char_ratio', 'devanagari_char_ratio', 'english_char_ratio'
        ]
        feature_names.extend(linguistic_feature_names)
        
        # 3. Extract stylometric features
        logger.info("Extracting stylometric features...")
        stylometric_features = self.extract_stylometric_features(texts)
        all_features.append(stylometric_features)
        stylometric_feature_names = [
            'avg_sentence_length', 'sentence_length_variance', 'function_word_ratio',
            'content_word_ratio', 'type_token_ratio', 'hapax_legomena_ratio',
            'readability_score', 'complexity_score'
        ]
        feature_names.extend(stylometric_feature_names)
        
        # 4. Extract psychological features
        logger.info("Extracting psychological features...")
        psychological_features = self.extract_psychological_features(texts)
        all_features.append(psychological_features)
        psychological_feature_names = [
            'emotion_word_ratio', 'positive_word_ratio', 'negative_word_ratio',
            'urgency_indicator_ratio', 'certainty_indicator_ratio', 'first_person_ratio',
            'second_person_ratio', 'third_person_ratio'
        ]
        feature_names.extend(psychological_feature_names)
        
        # Combine all features
        combined_features = np.hstack(all_features)
        
        logger.info(f"Total features extracted: {combined_features.shape}")
        logger.info(f"Feature breakdown: Embeddings={embeddings.shape[1] if include_embeddings else 0}, "
                   f"Linguistic={linguistic_features.shape[1]}, "
                   f"Stylometric={stylometric_features.shape[1]}, "
                   f"Psychological={psychological_features.shape[1]}")
        
        return combined_features, feature_names
    
    def save_features(self, features: np.ndarray, feature_names: List[str], 
                     filename: str):
        """Save extracted features to file"""
        np.savez(filename, 
                features=features, 
                feature_names=feature_names,
                model_name=self.model_name,
                max_length=self.max_length)
        logger.info(f"Features saved to {filename}")
    
    def load_features(self, filename: str) -> Tuple[np.ndarray, List[str]]:
        """Load features from file"""
        data = np.load(filename, allow_pickle=True)
        features = data['features']
        feature_names = data['feature_names'].tolist()
        logger.info(f"Features loaded from {filename}")
        return features, feature_names

# Example usage
if __name__ == "__main__":
    # Test the feature extractor
    extractor = MultilingualFeatureExtractor('bert-base-multilingual-cased')
    
    # Sample multilingual texts
    test_texts = [
        "ಇದು ಕನ್ನಡ ಭಾಷೆಯಲ್ಲಿನ ಸುದ್ದಿಯಾಗಿದೆ. ಇದು ನಿಜವಾದ ಮಾಹಿತಿಯನ್ನು ಹೊಂದಿದೆ.",
        "यह हिंदी भाषा में एक समाचार है। यह सत्य जानकारी प्रदान करता है।",
        "This is a news article in English. It contains factual information.",
        "ಈ ಸುದ್ದಿ ಸುಳ್ಳಾಗಿದೆ! ಇದನ್ನು ನಂಬಬೇಡಿ. ಇದು ಅಪಾಯಕಾರಿಯಾಗಿದೆ!!!"
    ]
    
    print("Testing Multilingual Feature Extractor:")
    print("=" * 50)
    
    # Extract all features
    features, feature_names = extractor.extract_all_features(
        test_texts, 
        include_embeddings=True, 
        batch_size=2
    )
    
    print(f"\nExtracted features shape: {features.shape}")
    print(f"Number of feature names: {len(feature_names)}")
    print(f"First 10 feature names: {feature_names[:10]}")
    print(f"Last 10 feature names: {feature_names[-10:]}")
    
    # Show some sample feature values
    print(f"\nSample feature values for first text:")
    print(f"Text length: {features[0][len(feature_names)-15]}")  # Approximate index for text_length
    print(f"Kannada char ratio: {features[0][len(feature_names)-3]}")  # Approximate index