"""
Kannada Text Preprocessing Module
================================

This module provides comprehensive preprocessing functions for Kannada text
including cleaning, tokenization, and normalization specifically designed
for fake news detection tasks.
"""

import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib

# Kannada-specific constants
KANNADA_UNICODE_RANGE = r'[\u0C80-\u0CFF]'
ENGLISH_PATTERN = r'[a-zA-Z]'
NUMBER_PATTERN = r'[0-9]'
PUNCTUATION_PATTERN = r'[^\w\s]'

# Common Kannada stop words (basic set)
KANNADA_STOP_WORDS = [
    'ಅದು', 'ಇದು', 'ಎಂದು', 'ಆದರೆ', 'ಅಥವಾ', 'ಮತ್ತು', 'ಅಥವ', 'ಅವರು', 
    'ನಾವು', 'ನಾನು', 'ನೀವು', 'ಅವರ', 'ನಮ್ಮ', 'ನನ್ನ', 'ನಿಮ್ಮ',
    'ಈ', 'ಆ', 'ಯಾವ', 'ಏನು', 'ಯಾರು', 'ಎಲ್ಲಿ', 'ಯಾಕೆ', 'ಹೇಗೆ',
    'ಮೇಲೆ', 'ಕೆಳಗೆ', 'ಮುಂದೆ', 'ಹಿಂದೆ', 'ಬಳಿ', 'ಪಕ್ಕ', 'ಅಲ್ಲಿ', 'ಇಲ್ಲಿ',
    'ಎಂಬ', 'ಎಂಬುದು', 'ಎಂದರೆ', 'ಅಂತ', 'ಆಗ', 'ಈಗ', 'ಮೊನ್ನೆ', 'ನಾಳೆ'
]

# English stop words to remove from mixed content
ENGLISH_STOP_WORDS = [
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
    'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have', 'had'
]

class KannadaPreprocessor:
    """
    Comprehensive preprocessing pipeline for Kannada text.
    """
    
    def __init__(self, remove_english=False, remove_numbers=True, 
                 remove_punctuation=True, remove_stop_words=True,
                 min_length=3, max_length=500):
        """
        Initialize the preprocessor with configuration options.
        
        Parameters:
        -----------
        remove_english : bool
            Whether to remove English characters
        remove_numbers : bool
            Whether to remove numbers
        remove_punctuation : bool
            Whether to remove punctuation
        remove_stop_words : bool
            Whether to remove stop words
        min_length : int
            Minimum text length to keep
        max_length : int
            Maximum text length to keep
        """
        self.remove_english = remove_english
        self.remove_numbers = remove_numbers
        self.remove_punctuation = remove_punctuation
        self.remove_stop_words = remove_stop_words
        self.min_length = min_length
        self.max_length = max_length
        
        # Combine stop words
        self.stop_words = set(KANNADA_STOP_WORDS + ENGLISH_STOP_WORDS)
        
        print(f"🔧 Initialized KannadaPreprocessor with:")
        print(f"   - Remove English: {remove_english}")
        print(f"   - Remove Numbers: {remove_numbers}")
        print(f"   - Remove Punctuation: {remove_punctuation}")
        print(f"   - Remove Stop Words: {remove_stop_words}")
        print(f"   - Text Length Range: {min_length}-{max_length}")
    
    def clean_text(self, text):
        """
        Clean individual text with various preprocessing steps.
        
        Parameters:
        -----------
        text : str
            Input text to clean
            
        Returns:
        --------
        str
            Cleaned text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to string and strip
        text = str(text).strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers if specified
        if self.remove_numbers:
            text = re.sub(NUMBER_PATTERN, '', text)
        
        # Remove English characters if specified
        if self.remove_english:
            text = re.sub(ENGLISH_PATTERN, '', text)
        
        # Remove punctuation if specified (but preserve Kannada punctuation marks)
        if self.remove_punctuation:
            # Remove English punctuation but keep some Kannada-specific marks
            english_punct = string.punctuation.replace('।', '').replace('॥', '')  # Keep Devanagari punctuation
            for punct in english_punct:
                text = text.replace(punct, ' ')
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """
        Simple tokenization for Kannada text.
        
        Parameters:
        -----------
        text : str
            Input text to tokenize
            
        Returns:
        --------
        list
            List of tokens
        """
        if not text:
            return []
        
        # Simple whitespace tokenization
        tokens = text.split()
        
        # Filter tokens based on length and content
        filtered_tokens = []
        for token in tokens:
            # Skip very short tokens
            if len(token) < 2:
                continue
            
            # Skip tokens that are only punctuation
            if re.match(r'^[^\w]+$', token, re.UNICODE):
                continue
            
            # Check if token has Kannada characters
            if re.search(KANNADA_UNICODE_RANGE, token):
                filtered_tokens.append(token)
            elif not self.remove_english and re.search(ENGLISH_PATTERN, token):
                # Keep English tokens if not removing them
                filtered_tokens.append(token.lower())
        
        return filtered_tokens
    
    def remove_stopwords(self, tokens):
        """
        Remove stop words from token list.
        
        Parameters:
        -----------
        tokens : list
            List of tokens
            
        Returns:
        --------
        list
            Filtered tokens without stop words
        """
        if not self.remove_stop_words:
            return tokens
        
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def preprocess_text(self, text):
        """
        Complete preprocessing pipeline for a single text.
        
        Parameters:
        -----------
        text : str
            Input text
            
        Returns:
        --------
        str
            Preprocessed text
        """
        # Clean text
        cleaned = self.clean_text(text)
        
        # Check length constraints
        if len(cleaned) < self.min_length or len(cleaned) > self.max_length:
            return ""
        
        # Tokenize
        tokens = self.tokenize(cleaned)
        
        # Remove stop words
        tokens = self.remove_stopwords(tokens)
        
        # Rejoin tokens
        processed_text = ' '.join(tokens)
        
        return processed_text
    
    def preprocess_dataframe(self, df, text_column='text'):
        """
        Preprocess all texts in a dataframe.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        text_column : str
            Name of the text column
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with preprocessed text
        """
        print(f"🔄 Preprocessing {len(df)} texts...")
        
        # Create a copy to avoid modifying original
        df_processed = df.copy()
        
        # Add preprocessing statistics
        df_processed['original_length'] = df_processed[text_column].str.len()
        
        # Apply preprocessing
        df_processed['processed_text'] = df_processed[text_column].apply(self.preprocess_text)
        
        # Add post-processing statistics
        df_processed['processed_length'] = df_processed['processed_text'].str.len()
        df_processed['length_reduction'] = df_processed['original_length'] - df_processed['processed_length']
        
        # Filter out empty texts
        original_count = len(df_processed)
        df_processed = df_processed[df_processed['processed_text'].str.len() > 0]
        filtered_count = len(df_processed)
        
        print(f"✅ Preprocessing completed!")
        print(f"   - Original texts: {original_count}")
        print(f"   - Valid texts after filtering: {filtered_count}")
        print(f"   - Filtered out: {original_count - filtered_count}")
        print(f"   - Average length reduction: {df_processed['length_reduction'].mean():.1f} chars")
        
        return df_processed
    
    def get_text_statistics(self, df, text_column='processed_text'):
        """
        Get detailed statistics about processed text.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with processed text
        text_column : str
            Name of the processed text column
            
        Returns:
        --------
        dict
            Dictionary with statistics
        """
        stats = {}
        
        # Basic statistics
        stats['total_texts'] = len(df)
        stats['avg_length'] = df[text_column].str.len().mean()
        stats['avg_word_count'] = df[text_column].str.split().str.len().mean()
        
        # Language statistics
        stats['kannada_texts'] = df[text_column].str.contains(KANNADA_UNICODE_RANGE, regex=True).sum()
        stats['mixed_language'] = df[text_column].str.contains(ENGLISH_PATTERN).sum()
        
        # Length distribution
        lengths = df[text_column].str.len()
        stats['length_stats'] = {
            'min': lengths.min(),
            'max': lengths.max(),
            'median': lengths.median(),
            'std': lengths.std()
        }
        
        return stats


def create_balanced_dataset(df, target_column='label', method='undersample', random_state=42):
    """
    Create a balanced dataset to handle class imbalance.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_column : str
        Name of the target column
    method : str
        Balancing method: 'undersample', 'oversample', or 'none'
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        Balanced dataframe
    """
    print(f"📊 Balancing dataset using method: {method}")
    
    # Check current distribution
    class_counts = df[target_column].value_counts()
    print(f"   Original distribution: {class_counts.to_dict()}")
    
    if method == 'none':
        return df
    
    elif method == 'undersample':
        # Undersample majority class
        min_class_size = class_counts.min()
        
        balanced_dfs = []
        for class_label in class_counts.index:
            class_df = df[df[target_column] == class_label]
            sampled_df = class_df.sample(n=min_class_size, random_state=random_state)
            balanced_dfs.append(sampled_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
    elif method == 'oversample':
        # Oversample minority class (simple duplication)
        max_class_size = class_counts.max()
        
        balanced_dfs = []
        for class_label in class_counts.index:
            class_df = df[df[target_column] == class_label]
            
            if len(class_df) < max_class_size:
                # Oversample by repeating samples
                n_repeats = max_class_size // len(class_df)
                remainder = max_class_size % len(class_df)
                
                repeated_df = pd.concat([class_df] * n_repeats, ignore_index=True)
                if remainder > 0:
                    extra_samples = class_df.sample(n=remainder, random_state=random_state)
                    repeated_df = pd.concat([repeated_df, extra_samples], ignore_index=True)
                
                balanced_dfs.append(repeated_df)
            else:
                balanced_dfs.append(class_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    # Shuffle the balanced dataset
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    final_counts = balanced_df[target_column].value_counts()
    print(f"   Balanced distribution: {final_counts.to_dict()}")
    
    return balanced_df


def create_train_test_split(df, text_column='processed_text', target_column='label',
                          test_size=0.2, random_state=42, stratify=True):
    """
    Create train-test split for the preprocessed data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed dataframe
    text_column : str
        Name of the text column
    target_column : str
        Name of the target column
    test_size : float
        Proportion of test set
    random_state : int
        Random state for reproducibility
    stratify : bool
        Whether to stratify the split
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    print(f"🔀 Creating train-test split (test_size={test_size})")
    
    X = df[text_column]
    y = df[target_column]
    
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )
    
    print(f"✅ Split completed:")
    print(f"   - Training samples: {len(X_train)}")
    print(f"   - Test samples: {len(X_test)}")
    print(f"   - Training label distribution: {y_train.value_counts().to_dict()}")
    print(f"   - Test label distribution: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test


def save_preprocessed_data(X_train, X_test, y_train, y_test, output_dir='data'):
    """
    Save the preprocessed and split data.
    
    Parameters:
    -----------
    X_train, X_test, y_train, y_test : arrays
        Split data
    output_dir : str
        Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save as joblib for efficiency
    joblib.dump(X_train, output_path / 'X_train.joblib')
    joblib.dump(X_test, output_path / 'X_test.joblib')
    joblib.dump(y_train, output_path / 'y_train.joblib')
    joblib.dump(y_test, output_path / 'y_test.joblib')
    
    # Also save as CSV for inspection
    train_df = pd.DataFrame({'text': X_train, 'label': y_train})
    test_df = pd.DataFrame({'text': X_test, 'label': y_test})
    
    train_df.to_csv(output_path / 'train_processed.csv', index=False, encoding='utf-8')
    test_df.to_csv(output_path / 'test_processed.csv', index=False, encoding='utf-8')
    
    print(f"💾 Preprocessed data saved to {output_path}")
    print(f"   - Training set: {len(X_train)} samples")
    print(f"   - Test set: {len(X_test)} samples")


if __name__ == "__main__":
    # Load the processed dataset from data exploration
    print("🔄 Loading dataset for preprocessing...")
    df = pd.read_csv('data/processed_dataset.csv', encoding='utf-8')
    
    # Initialize preprocessor
    preprocessor = KannadaPreprocessor(
        remove_english=False,  # Keep English for mixed content
        remove_numbers=True,
        remove_punctuation=True,
        remove_stop_words=True,
        min_length=10,  # Minimum 10 characters
        max_length=200  # Maximum 200 characters
    )
    
    # Preprocess the data
    df_processed = preprocessor.preprocess_dataframe(df, text_column='text')
    
    # Get statistics
    stats = preprocessor.get_text_statistics(df_processed)
    print(f"\n📊 Preprocessing Statistics:")
    print(f"   - Total texts: {stats['total_texts']}")
    print(f"   - Average length: {stats['avg_length']:.1f} characters")
    print(f"   - Average word count: {stats['avg_word_count']:.1f} words")
    print(f"   - Kannada texts: {stats['kannada_texts']} ({stats['kannada_texts']/stats['total_texts']*100:.1f}%)")
    print(f"   - Mixed language: {stats['mixed_language']} ({stats['mixed_language']/stats['total_texts']*100:.1f}%)")
    
    # Handle class imbalance: oversample to improve recall on minority (fake)
    df_balanced = create_balanced_dataset(df_processed, method='oversample')
    
    # Create train-test split
    X_train, X_test, y_train, y_test = create_train_test_split(
        df_balanced, 
        text_column='processed_text',
        target_column='label'
    )
    
    # Save preprocessed data
    save_preprocessed_data(X_train, X_test, y_train, y_test)
    
    print("\n🎉 Preprocessing completed successfully!")