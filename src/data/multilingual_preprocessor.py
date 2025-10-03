"""
Multilingual Text Preprocessor for Fake News Detection
Supports Kannada, English, and Hindi languages as specified in the project report
"""

import re
import string
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from langdetect import detect, LangDetectError
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import html

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultilingualPreprocessor:
    """
    Advanced multilingual text preprocessor specifically designed for
    Kannada fake news detection system as per project requirements
    """
    
    def __init__(self):
        self.supported_languages = ['kn', 'en', 'hi']  # Kannada, English, Hindi
        self.language_names = {
            'kn': 'Kannada',
            'en': 'English', 
            'hi': 'Hindi'
        }
        
        # Initialize NLTK components
        self._download_nltk_data()
        
        # Kannada script range (Unicode)
        self.kannada_range = r'[\u0C80-\u0CFF]'
        # Devanagari script range for Hindi
        self.devanagari_range = r'[\u0900-\u097F]'
        
        # Load stopwords for supported languages
        self.stopwords_dict = self._load_multilingual_stopwords()
        
        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.phone_pattern = re.compile(r'[\+]?[1-9]?[0-9]{7,15}')
        self.html_tag_pattern = re.compile(r'<.*?>')
        
    def _download_nltk_data(self):
        """Download required NLTK data"""
        nltk_data = ['punkt', 'stopwords', 'wordnet']
        for data in nltk_data:
            try:
                nltk.data.find(f'tokenizers/{data}')
            except LookupError:
                nltk.download(data, quiet=True)
                
    def _load_multilingual_stopwords(self) -> Dict[str, set]:
        """Load stopwords for supported languages"""
        stopwords_dict = {}
        
        # English stopwords
        try:
            stopwords_dict['en'] = set(stopwords.words('english'))
        except:
            stopwords_dict['en'] = set()
            
        # Basic Kannada stopwords (common ones)
        stopwords_dict['kn'] = {
            'ಅದು', 'ಇದು', 'ಆ', 'ಈ', 'ಒಂದು', 'ಅವರು', 'ಅವನು', 'ಅವಳು',
            'ನಾನು', 'ನೀವು', 'ನಾವು', 'ಅವರ', 'ಮತ್ತು', 'ಅಥವಾ', 'ಆದರೆ',
            'ಎಂದು', 'ಇಲ್ಲಿ', 'ಅಲ್ಲಿ', 'ಯಾವ', 'ಯಾರು', 'ಏನು', 'ಎಲ್ಲಿ',
            'ಎಷ್ಟು', 'ಹೇಗೆ', 'ಯಾವಾಗ', 'ಇದೆ', 'ಇಲ್ಲ', 'ಅವು', 'ಅದನ್ನು',
            'ಇದನ್ನು', 'ಆ', 'ಈ', 'ಮೊದಲು', 'ನಂತರ', 'ಮೇಲೆ', 'ಕೆಳಗೆ'
        }
        
        # Basic Hindi stopwords
        stopwords_dict['hi'] = {
            'और', 'का', 'एक', 'को', 'ही', 'में', 'है', 'की', 'से', 'यह',
            'वह', 'पर', 'या', 'हो', 'था', 'थी', 'थे', 'गई', 'हैं', 'कि',
            'जो', 'कर', 'लिए', 'अपने', 'हुआ', 'हुई', 'हुए', 'इस', 'उस',
            'तक', 'साथ', 'बाद', 'फिर', 'यदि', 'जब', 'तब', 'कहा', 'गया'
        }
        
        return stopwords_dict
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of input text with fallback mechanisms
        """
        if not text or len(text.strip()) < 3:
            return 'unknown'
            
        # First check for script patterns
        kannada_chars = len(re.findall(self.kannada_range, text))
        devanagari_chars = len(re.findall(self.devanagari_range, text))
        total_chars = len(re.sub(r'[^\w]', '', text))
        
        if total_chars > 0:
            kannada_ratio = kannada_chars / total_chars
            devanagari_ratio = devanagari_chars / total_chars
            
            if kannada_ratio > 0.3:
                return 'kn'
            elif devanagari_ratio > 0.3:
                return 'hi'
        
        # Fallback to langdetect
        try:
            detected = detect(text)
            if detected in self.supported_languages:
                return detected
            elif detected in ['mr', 'bn', 'gu']:  # Related Indic languages
                return 'hi'
            else:
                return 'en'  # Default to English
        except LangDetectError:
            return 'en'
    
    def clean_text(self, text: str, language: str = None) -> str:
        """
        Clean and normalize text based on language requirements
        """
        if not isinstance(text, str):
            return ""
        
        if language is None:
            language = self.detect_language(text)
        
        # Remove HTML entities and tags
        text = html.unescape(text)
        text = self.html_tag_pattern.sub(' ', text)
        
        # Remove URLs, emails, phone numbers
        text = self.url_pattern.sub(' ', text)
        text = self.email_pattern.sub(' ', text)
        text = self.phone_pattern.sub(' ', text)
        
        # Language-specific cleaning
        if language == 'kn':
            text = self._clean_kannada_text(text)
        elif language == 'hi':
            text = self._clean_hindi_text(text)
        else:  # Default English cleaning
            text = self._clean_english_text(text)
        
        # Common cleaning steps
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def _clean_kannada_text(self, text: str) -> str:
        """Kannada-specific text cleaning"""
        # Remove English mixed with Kannada if it's less than 10%
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = len(re.sub(r'[^\w]', '', text))
        
        if total_chars > 0 and english_chars / total_chars < 0.1:
            text = re.sub(r'[a-zA-Z]+', ' ', text)
        
        # Keep Kannada characters, numbers, and basic punctuation
        text = re.sub(r'[^\u0C80-\u0CFF\s\d.,!?;:\-\'\"]', ' ', text)
        
        return text
    
    def _clean_hindi_text(self, text: str) -> str:
        """Hindi-specific text cleaning"""
        # Keep Devanagari characters, numbers, and basic punctuation
        text = re.sub(r'[^\u0900-\u097F\s\d.,!?;:\-\'\"]', ' ', text)
        
        return text
    
    def _clean_english_text(self, text: str) -> str:
        """English-specific text cleaning"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:\-\'\"]', ' ', text)
        
        return text
    
    def remove_stopwords(self, text: str, language: str = None) -> str:
        """
        Remove stopwords based on detected or specified language
        """
        if language is None:
            language = self.detect_language(text)
        
        if language not in self.stopwords_dict:
            language = 'en'  # Default to English
        
        # Tokenize text
        try:
            if language in ['kn', 'hi']:
                # Simple whitespace tokenization for Indic languages
                words = text.split()
            else:
                words = word_tokenize(text.lower())
        except:
            words = text.split()
        
        # Remove stopwords
        stopwords_set = self.stopwords_dict[language]
        filtered_words = [word for word in words if word not in stopwords_set]
        
        return ' '.join(filtered_words)
    
    def tokenize_multilingual(self, text: str, language: str = None) -> List[str]:
        """
        Tokenize text based on language
        """
        if language is None:
            language = self.detect_language(text)
        
        try:
            if language in ['kn', 'hi']:
                # For Indic languages, use simple word splitting
                # as NLTK tokenizers may not work well
                return text.split()
            else:
                return word_tokenize(text)
        except:
            return text.split()
    
    def extract_language_features(self, text: str) -> Dict[str, float]:
        """
        Extract language-specific features for classification
        """
        features = {}
        
        # Basic text statistics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['char_count'] = len(text)
        features['sentence_count'] = len(sent_tokenize(text)) if text else 0
        
        # Language distribution
        kannada_chars = len(re.findall(self.kannada_range, text))
        devanagari_chars = len(re.findall(self.devanagari_range, text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = len(re.sub(r'[^\w]', '', text))
        
        if total_chars > 0:
            features['kannada_ratio'] = kannada_chars / total_chars
            features['devanagari_ratio'] = devanagari_chars / total_chars
            features['english_ratio'] = english_chars / total_chars
        else:
            features['kannada_ratio'] = 0
            features['devanagari_ratio'] = 0
            features['english_ratio'] = 0
        
        # Punctuation and special characters
        punctuation_count = sum([1 for char in text if char in string.punctuation])
        features['punctuation_ratio'] = punctuation_count / len(text) if text else 0
        
        # Uppercase ratio (mainly for English/mixed content)
        upper_count = sum([1 for char in text if char.isupper()])
        features['uppercase_ratio'] = upper_count / len(text) if text else 0
        
        # Digit ratio
        digit_count = sum([1 for char in text if char.isdigit()])
        features['digit_ratio'] = digit_count / len(text) if text else 0
        
        # Average word length
        words = text.split()
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        
        return features
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'text', 
                           label_column: str = 'label') -> pd.DataFrame:
        """
        Preprocess entire dataframe with multilingual support
        """
        logger.info(f"Preprocessing dataframe with {len(df)} rows")
        
        # Make a copy to avoid modifying original
        processed_df = df.copy()
        
        # Detect languages
        logger.info("Detecting languages...")
        processed_df['detected_language'] = processed_df[text_column].apply(
            lambda x: self.detect_language(str(x))
        )
        
        # Clean text
        logger.info("Cleaning text...")
        processed_df['cleaned_text'] = processed_df.apply(
            lambda row: self.clean_text(str(row[text_column]), row['detected_language']), 
            axis=1
        )
        
        # Remove stopwords
        logger.info("Removing stopwords...")
        processed_df['text_no_stopwords'] = processed_df.apply(
            lambda row: self.remove_stopwords(row['cleaned_text'], row['detected_language']), 
            axis=1
        )
        
        # Extract language features
        logger.info("Extracting language features...")
        language_features = processed_df['cleaned_text'].apply(self.extract_language_features)
        feature_df = pd.DataFrame(list(language_features))
        
        # Combine with original dataframe
        result_df = pd.concat([processed_df, feature_df], axis=1)
        
        # Filter out very short texts
        min_length = 10
        result_df = result_df[result_df['text_length'] >= min_length]
        
        logger.info(f"Preprocessing complete. Final dataset size: {len(result_df)} rows")
        
        return result_df
    
    def validate_language_support(self, text_series: pd.Series) -> Dict[str, int]:
        """
        Validate language distribution in dataset
        """
        language_counts = {}
        
        for text in text_series:
            lang = self.detect_language(str(text))
            language_counts[lang] = language_counts.get(lang, 0) + 1
        
        total = len(text_series)
        logger.info("Language distribution:")
        for lang, count in language_counts.items():
            lang_name = self.language_names.get(lang, lang)
            percentage = (count / total) * 100
            logger.info(f"  {lang_name} ({lang}): {count} texts ({percentage:.1f}%)")
        
        return language_counts

# Example usage and testing
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = MultilingualPreprocessor()
    
    # Test with sample texts
    test_texts = [
        "ಇದು ಕನ್ನಡ ಭಾಷೆಯ ಪರೀಕ್ಷೆಯಾಗಿದೆ. ಈ ವಾಕ್ಯವನ್ನು ಸ್ವಚ್ಛಗೊಳಿಸಲಾಗುತ್ತದೆ.",  # Kannada
        "यह हिंदी भाषा का परीक्षण है। इस वाक्य को साफ किया जाएगा।",  # Hindi
        "This is an English language test. This sentence will be cleaned.",  # English
        "मिक्स्ड content with English और हिंदी mixed together ಮತ್ತು ಕನ್ನಡ"  # Mixed
    ]
    
    print("Testing Multilingual Preprocessor:")
    print("=" * 50)
    
    for i, text in enumerate(test_texts):
        print(f"\nTest {i+1}:")
        print(f"Original: {text}")
        
        # Detect language
        lang = preprocessor.detect_language(text)
        print(f"Detected Language: {preprocessor.language_names.get(lang, lang)}")
        
        # Clean text
        cleaned = preprocessor.clean_text(text)
        print(f"Cleaned: {cleaned}")
        
        # Remove stopwords
        no_stopwords = preprocessor.remove_stopwords(cleaned, lang)
        print(f"No Stopwords: {no_stopwords}")
        
        # Extract features
        features = preprocessor.extract_language_features(text)
        print(f"Key Features: Length={features['text_length']}, "
              f"Kannada={features['kannada_ratio']:.2f}, "
              f"English={features['english_ratio']:.2f}")