"""
Global configuration for Multilingual Fake News Detection (report-aligned)
"""

# Default model alias: 'mbert' or 'xlm_roberta'
MODEL_DEFAULT = 'mbert'

# Map aliases to Hugging Face model names
MODEL_MAP = {
    'mbert': 'bert-base-multilingual-cased',
    'xlm_roberta': 'xlm-roberta-base'
}

# Sequence length for transformer tokenization (reduced for latency)
MAX_LENGTH = 256

# Training defaults
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5

# Targets per report
PERFORMANCE_TARGET = 2.0  # seconds per article
ACCURACY_TARGET = 0.90     # 90%+

# Languages supported
SUPPORTED_LANGUAGES = ['kn', 'en', 'hi']  # Kannada, English, Hindi

# API defaults
API_PORT = 5000
SECRET_KEY = 'dev-secret-key'
DEBUG = False
