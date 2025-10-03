#!/usr/bin/env python3
"""
Main Application for Multilingual Fake News Detection System
Kannada Language Support with English and Hindi

This is the main entry point for the fake news detection system as specified
in the project report from Sahyadri College of Engineering & Management.

Authors:
- MUHAMMED AQUIF (4SF22CS114)
- PRATHAM AMIN (4SF22CS145) 
- VISHWA (4SF22CS247)
- B M SHASHANK (4SF23CS402)

Guide: Mrs. SUKETHA, Assistant Professor, Dept of CSE, SCEM
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our modules
from src.data.multilingual_preprocessor import MultilingualPreprocessor
from src.models.multilingual_feature_extractor import MultilingualFeatureExtractor
from src.models.multilingual_classifier import MultilingualFakeNewsDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fake_news_detection.log')
    ]
)
logger = logging.getLogger(__name__)

class FakeNewsDetectionSystem:
    """
    Main class for the Multilingual Fake News Detection System
    """
    
    def __init__(self):
        """Initialize the system components"""
        logger.info("=" * 60)
        logger.info("MULTILINGUAL FAKE NEWS DETECTION SYSTEM")
        logger.info("Kannada Language Support with English and Hindi")
        logger.info("=" * 60)
        logger.info("Team: AQUIF, PRATHAM, VISHWA, SHASHANK")
        logger.info("Guide: Mrs. SUKETHA, SCEM")
        logger.info("=" * 60)
        
        # Initialize components
        self.preprocessor = None
        self.feature_extractor = None
        self.detector = None
        
        # Configuration
        self.model_configs = {
            'mbert': 'bert-base-multilingual-cased',
            'xlm_roberta': 'xlm-roberta-base'
        }
        
        self.supported_languages = ['kn', 'en', 'hi']  # Kannada, English, Hindi
        
        # Fallback classic ML components (per report recommendation)
        self.tfidf_vectorizer = None
        self.lr_model = None
        
    def initialize_components(self, model_type: str = 'mbert'):
        """
        Initialize system components
        
        Args:
            model_type: Type of transformer model to use ('mbert' or 'xlm_roberta')
        """
        try:
            logger.info(f"Initializing system components with {model_type}...")
            
            # Initialize preprocessor
            logger.info("Loading multilingual preprocessor...")
            self.preprocessor = MultilingualPreprocessor()
            logger.info("✓ Preprocessor loaded successfully")
            
            # Initialize feature extractor
            logger.info(f"Loading {model_type} feature extractor...")
            model_name = self.model_configs.get(model_type, self.model_configs['mbert'])
            self.feature_extractor = MultilingualFeatureExtractor(
                model_name=model_name,
                max_length=512
            )
            logger.info("✓ Feature extractor loaded successfully")
            
            # Initialize detector
            logger.info("Loading fake news detector...")
            self.detector = MultilingualFakeNewsDetector(
                model_name=model_name,
                max_length=512
            )
            logger.info("✓ Detector initialized successfully")
            
            logger.info("🎉 All components initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load and preprocess data
        
        Args:
            data_path: Path to the dataset CSV file
            
        Returns:
            Processed DataFrame
        """
        try:
            logger.info(f"Loading data from {data_path}...")
            
            # Check if file exists
            if not os.path.exists(data_path):
                logger.error(f"Data file not found: {data_path}")
                return None
            
            # Load data
            df = pd.read_csv(data_path)
            logger.info(f"Loaded {len(df)} samples from dataset")
            
            # Validate data structure
            required_columns = ['text', 'label']
            for col in required_columns:
                if col not in df.columns:
                    logger.error(f"Required column '{col}' not found in dataset")
                    return None
            
            # Basic data statistics
            logger.info("Dataset Statistics:")
            logger.info(f"  Total samples: {len(df)}")
            logger.info(f"  Real news: {len(df[df['label'] == 0])}")
            logger.info(f"  Fake news: {len(df[df['label'] == 1])}")
            
            # Language distribution
            if self.preprocessor:
                language_dist = self.preprocessor.validate_language_support(df['text'])
                logger.info("Language distribution validated ✓")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        if self.preprocessor is None:
            logger.error("Preprocessor not initialized!")
            return None
        
        try:
            logger.info("Starting data preprocessing...")
            processed_df = self.preprocessor.preprocess_dataframe(
                df, text_column='text', label_column='label'
            )
            logger.info("✓ Data preprocessing completed successfully")
            return processed_df
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            return None
    
    def extract_features(self, texts: List[str]) -> tuple:
        """
        Extract features from texts
        
        Args:
            texts: List of text strings
            
        Returns:
            Tuple of (features, feature_names)
        """
        if self.feature_extractor is None:
            logger.error("Feature extractor not initialized!")
            return None, None
        
        try:
            logger.info("Extracting features...")
            features, feature_names = self.feature_extractor.extract_all_features(
                texts, include_embeddings=True, batch_size=16
            )
            logger.info("✓ Feature extraction completed successfully")
            return features, feature_names
            
        except Exception as e:
            logger.error(f"Error in feature extraction: {str(e)}")
            return None, None
    
    def train_model(self, df: pd.DataFrame, **train_params) -> Dict:
        """
        Train the fake news detection model
        
        Args:
            df: Processed DataFrame
            **train_params: Training parameters
            
        Returns:
            Training results dictionary
        """
        if self.detector is None:
            logger.error("Detector not initialized!")
            return None
        
        try:
            logger.info("Starting model training...")
            
            # Prepare data
            texts = df['cleaned_text'].tolist() if 'cleaned_text' in df.columns else df['text'].tolist()
            labels = df['label'].tolist()
            
            # Default training parameters
            default_params = {
                'batch_size': 16,
                'epochs': 3,
                'learning_rate': 2e-5,
                'test_size': 0.2
            }
            default_params.update(train_params)
            
            # Prepare datasets
            train_dataset, test_dataset, X_test, y_test = self.detector.prepare_data(
                texts, labels, 
                test_size=default_params['test_size'],
                random_state=42
            )
            
            # Train model
            training_history = self.detector.train(
                train_dataset, 
                test_dataset,
                batch_size=default_params['batch_size'],
                epochs=default_params['epochs'],
                learning_rate=default_params['learning_rate'],
                save_path='models/multilingual_fake_news_model.pth'
            )
            
            # Evaluate model
            results = self.detector.evaluate(X_test, y_test)
            
            logger.info("✓ Model training completed successfully")
            logger.info(f"Final Accuracy: {results['accuracy']:.4f}")
            logger.info(f"Final F1-Score: {results['f1_score']:.4f}")
            
            return {
                'training_history': training_history,
                'evaluation_results': results,
                'test_texts': X_test,
                'test_labels': y_test
            }
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            return None
    
    def predict_single(self, text: str) -> Dict:
        """
        Make prediction on a single text
        
        Args:
            text: Input text string
            
        Returns:
            Prediction results dictionary
        """
        if self.detector is None or self.preprocessor is None:
            logger.error("System not properly initialized!")
            return None
        
        try:
            # Preprocess text
            language = self.preprocessor.detect_language(text)
            cleaned_text = self.preprocessor.clean_text(text, language)
            
            prediction = None
            confidence = None
            backend = None
            
            # Primary: transformer model if available
            if self.detector.model is not None:
                try:
                    predictions, confidences = self.detector.predict([cleaned_text])
                    prediction = int(predictions[0])
                    confidence = float(confidences[0])
                    backend = 'transformer'
                except Exception as e:
                    logger.warning(f"Transformer prediction failed: {e}")
                    prediction = None
                    confidence = None
            
            # Fallback: TF-IDF + Logistic Regression as per report
            if prediction is None:
                # Lazy-load fallback models
                if self.tfidf_vectorizer is None or self.lr_model is None:
                    try:
                        vec_path = Path('models') / 'tfidf_vectorizer.joblib'
                        lr_path = Path('models') / 'logistic_regression.joblib'
                        if vec_path.exists() and lr_path.exists():
                            self.tfidf_vectorizer = joblib.load(vec_path)
                            self.lr_model = joblib.load(lr_path)
                            logger.info("Loaded TF-IDF + Logistic Regression fallback models")
                        else:
                            logger.warning("Fallback TF-IDF artifacts not found in models/; cannot predict")
                    except Exception as e:
                        logger.warning(f"Failed to load fallback models: {e}")
                
                if self.tfidf_vectorizer is not None and self.lr_model is not None:
                    try:
                        X = self.tfidf_vectorizer.transform([cleaned_text])
                        pred = int(self.lr_model.predict(X)[0])
                        proba = self.lr_model.predict_proba(X)[0] if hasattr(self.lr_model, 'predict_proba') else None
                        conf = float(proba[pred]) if proba is not None else 0.75
                        prediction = pred
                        confidence = conf
                        backend = 'tfidf_lr'
                    except Exception as e:
                        logger.warning(f"TF-IDF fallback prediction failed: {e}")
                        prediction = None
                        confidence = None
            
            # Extract features
            features = self.preprocessor.extract_language_features(text)
            
            return {
                'original_text': text,
                'cleaned_text': cleaned_text,
                'detected_language': language,
                'language_name': self.preprocessor.language_names.get(language, language),
                'prediction': prediction,
                'confidence': confidence,
                'prediction_label': 'Fake' if prediction == 1 else 'Real' if prediction == 0 else 'Unknown',
                'backend': backend,
                'features': features
            }
            
        except Exception as e:
            logger.error(f"Error in single prediction: {str(e)}")
            return None
    
    def run_demo(self):
        """Run a demonstration of the system"""
        logger.info("\n" + "=" * 50)
        logger.info("RUNNING SYSTEM DEMONSTRATION")
        logger.info("=" * 50)
        
        # Sample multilingual texts
        demo_texts = [
            {
                'text': 'ಇದು ನಿಜವಾದ ಸುದ್ದಿ ವರದಿಯಾಗಿದೆ. ಸರ್ಕಾರವು ಹೊಸ ನೀತಿಯನ್ನು ಘೋಷಿಸಿದೆ.',
                'expected': 'Real',
                'language': 'Kannada'
            },
            {
                'text': 'ಈ ಸುದ್ದಿ ಸಂಪೂರ್ಣವಾಗಿ ಸುಳ್ಳು! ನಂಬಬೇಡಿ!!! ಇದು ಅಪಾಯಕಾರಿ!!!',
                'expected': 'Fake',
                'language': 'Kannada'
            },
            {
                'text': 'This is a legitimate news report about recent government policy changes and their implementation.',
                'expected': 'Real',
                'language': 'English'
            },
            {
                'text': 'BREAKING NEWS: Shocking discovery!!! Scientists dont want you to know this!!! Click now for amazing results!!!',
                'expected': 'Fake',
                'language': 'English'
            },
            {
                'text': 'सरकार ने नई नीति की घोषणा की है। यह महत्वपूर्ण समाचार है जो नागरिकों के लिए फायदेमंद है।',
                'expected': 'Real',
                'language': 'Hindi'
            },
            {
                'text': 'यह पूरी तरह से झूठी खबर है! इसे बिल्कुल मत मानिए!!! यह बहुत खतरनाक है!!!',
                'expected': 'Fake',
                'language': 'Hindi'
            }
        ]
        
        for i, demo in enumerate(demo_texts, 1):
            logger.info(f"\n--- Demo Text {i} ({demo['language']}) ---")
            logger.info(f"Text: {demo['text']}")
            logger.info(f"Expected: {demo['expected']}")
            
            result = self.predict_single(demo['text'])
            if result:
                logger.info(f"Detected Language: {result['language_name']}")
                logger.info(f"Prediction: {result['prediction_label']}")
                if result['confidence']:
                    logger.info(f"Confidence: {result['confidence']:.3f}")
                logger.info(f"Text Length: {result['features']['text_length']} chars")
                logger.info(f"Language Ratios - Kannada: {result['features']['kannada_ratio']:.2f}, "
                          f"English: {result['features']['english_ratio']:.2f}, "
                          f"Hindi: {result['features']['devanagari_ratio']:.2f}")
            else:
                logger.error("Failed to process text")
        
        logger.info("\n" + "=" * 50)
        logger.info("DEMONSTRATION COMPLETED")
        logger.info("=" * 50)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Multilingual Fake News Detection System - Kannada Language Support'
    )
    
    parser.add_argument(
        '--mode', 
        choices=['demo', 'train', 'api', 'predict', 'crossval'],
        default='demo',
        help='Mode to run the system in'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        help='Path to dataset CSV file'
    )
    
    parser.add_argument(
        '--text',
        type=str,
        help='Single text to predict (for predict mode)'
    )
    
    parser.add_argument(
        '--model',
        choices=['mbert', 'xlm_roberta'],
        default='mbert',
        help='Transformer model to use'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Training batch size'
    )
    
    parser.add_argument(
        '--k_folds',
        type=int,
        default=5,
        help='Number of folds for cross-validation'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port for API mode'
    )
    
    args = parser.parse_args()
    
    # Initialize system
    system = FakeNewsDetectionSystem()
    system.initialize_components(model_type=args.model)
    
    if args.mode == 'demo':
        # Run demonstration
        system.run_demo()
        
    elif args.mode == 'train':
        # Train model
        if not args.data:
            logger.error("--data argument required for training mode")
            return
        
        # Load data
        df = system.load_data(args.data)
        if df is None:
            return
        
        # Preprocess data
        processed_df = system.preprocess_data(df)
        if processed_df is None:
            return
        
        # Train model
        train_params = {
            'epochs': args.epochs,
            'batch_size': args.batch_size
        }
        
        results = system.train_model(processed_df, **train_params)
        if results:
            logger.info("Training completed successfully!")
            logger.info("Model saved to models/multilingual_fake_news_model.pth")
        
    elif args.mode == 'predict':
        # Single prediction
        if not args.text:
            logger.error("--text argument required for predict mode")
            return
        
        result = system.predict_single(args.text)
        if result:
            print("\n" + "=" * 50)
            print("PREDICTION RESULTS")
            print("=" * 50)
            print(f"Text: {result['original_text']}")
            print(f"Language: {result['language_name']}")
            print(f"Prediction: {result['prediction_label']}")
            if result['confidence']:
                print(f"Confidence: {result['confidence']:.3f}")
            print("=" * 50)
        
    elif args.mode == 'api':
        # Start API server
        logger.info("Starting API server...")
        from src.api.fake_news_api import app
        
        app.run(
            host='0.0.0.0',
            port=args.port,
            debug=False,
            threaded=True
        )
    
    elif args.mode == 'crossval':
        # K-fold cross-validation according to report
        if not args.data:
            logger.error("--data argument required for crossval mode")
            return
        
        # Load data
        df = system.load_data(args.data)
        if df is None:
            return
        
        # Preprocess data
        processed_df = system.preprocess_data(df)
        if processed_df is None:
            return
        
        texts = processed_df['cleaned_text'].tolist() if 'cleaned_text' in processed_df.columns else processed_df['text'].tolist()
        labels = processed_df['label'].tolist()
        
        # Perform cross-validation
        cv_results = system.detector.cross_validate(
            texts, labels,
            k_folds=args.k_folds,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=2e-5
        )
        
        # Save results
        Path('results').mkdir(exist_ok=True)
        with open('results/crossval_results.json', 'w', encoding='utf-8') as f:
            json.dump(cv_results, f, ensure_ascii=False, indent=2)
        
        logger.info("Cross-validation results saved to results/crossval_results.json")
    
    else:
        logger.error(f"Unknown mode: {args.mode}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("System interrupted by user")
    except Exception as e:
        logger.error(f"System error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())