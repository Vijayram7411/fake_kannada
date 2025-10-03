"""
Configuration settings for the Fake News Detection system
"""
import os
from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model configurations
MODEL_CONFIGS = {
    'mbert': {
        'model_name': 'bert-base-multilingual-cased',
        'max_length': 512,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'epochs': 3
    },
    'xlm_roberta': {
        'model_name': 'xlm-roberta-base',
        'max_length': 512,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'epochs': 3
    }
}

# Language settings
SUPPORTED_LANGUAGES = ['kn', 'en', 'hi']  # Kannada, English, Hindi
LANGUAGE_NAMES = {
    'kn': 'Kannada',
    'en': 'English', 
    'hi': 'Hindi'
}

# Data collection settings
DATA_SOURCES = {
    'kannada_news_sites': [
        'https://www.prajavani.net/',
        'https://www.vijaykarnataka.com/',
        'https://www.kannadaprabha.com/',
        'https://www.udayavani.com/'
    ],
    'english_news_sites': [
        'https://www.deccanherald.com/',
        'https://www.thehindu.com/',
        'https://timesofindia.indiatimes.com/'
    ],
    'fact_check_sites': [
        'https://www.altnews.in/',
        'https://factchecker.in/',
        'https://www.boomlive.in/'
    ]
}

# Training parameters
TRAINING_CONFIG = {
    'test_size': 0.2,
    'validation_size': 0.1,
    'random_state': 42,
    'k_fold': 5,
    'shuffle': True
}

# Performance requirements (from report)
PERFORMANCE_REQUIREMENTS = {
    'response_time_target': 2.0,  # seconds
    'accuracy_target': 0.90,     # 90%
    'throughput_target': 1000    # articles per hour
}

# API settings
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': False,
    'max_content_length': 16 * 1024 * 1024  # 16MB
}

# Security settings
SECURITY_CONFIG = {
    'secret_key': os.environ.get('SECRET_KEY', 'dev-key-change-in-production'),
    'jwt_secret_key': os.environ.get('JWT_SECRET_KEY', 'jwt-dev-key'),
    'encrypt_data': True
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': 'logs/app.log',
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': False
        }
    }
}

# Feature extraction settings
FEATURE_CONFIG = {
    'complexity_features': True,
    'stylometric_features': True,
    'psychological_features': True,
    'use_embeddings': True,
    'embedding_dim': 768
}

# Text preprocessing settings
PREPROCESSING_CONFIG = {
    'remove_html': True,
    'remove_urls': True,
    'remove_special_chars': True,
    'convert_to_lowercase': True,
    'remove_stop_words': True,
    'apply_stemming': False,  # Better to use full words for transformer models
    'min_text_length': 10,
    'max_text_length': 5000
}