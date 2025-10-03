# Multilingual Fake News Detection System - Deployment Guide

## 🌟 Project Overview

This is the **Multilingual Fake News Detection System** developed for Kannada language support, along with English and Hindi languages, as specified in the project report from Sahyadri College of Engineering & Management.

**Team Members:**
- MUHAMMED AQUIF (4SF22CS114)
- PRATHAM AMIN (4SF22CS145) 
- VISHWA (4SF22CS247)
- B M SHASHANK (4SF23CS402)

**Guide:** Mrs. SUKETHA, Assistant Professor, Dept of CSE, SCEM

## 🎯 System Features

### Core Features (As per Project Report)
- ✅ Multilingual NLP support (Kannada, English, Hindi)
- ✅ Transformer-based models (mBERT, XLM-RoBERTa)
- ✅ Real-time classification with <2 second response time target
- ✅ 90%+ accuracy target on benchmark datasets
- ✅ REST API for integration with external platforms
- ✅ Web-based user interface
- ✅ Comprehensive feature extraction (Complexity, Stylometric, Psychological)
- ✅ K-fold cross-validation training
- ✅ Performance monitoring and statistics

### Technical Implementation
- **Architecture:** Transformer-based neural networks
- **Models:** Multilingual BERT (mBERT), XLM-RoBERTa
- **Languages:** Kannada (ಕನ್ನಡ), English, Hindi (हिंदी)
- **Framework:** PyTorch, Transformers, Flask
- **Evaluation:** Accuracy, Precision, Recall, F1-Score

## 🚀 Quick Start

### Method 1: Run Demo (Recommended for first-time users)
```bash
# Navigate to project directory
cd C:\fakee

# Run the demonstration
python main.py --mode demo
```

### Method 2: Start Web Interface
```bash
# Start the web API server
python main.py --mode api --port 5000

# Open browser and go to: http://localhost:5000
```

### Method 3: Command Line Prediction
```bash
# Predict a single text
python main.py --mode predict --text "ಇದು ಸುಳ್ಳು ಸುದ್ದಿಯಾಗಿದೆ!"
```

## 📋 Prerequisites

### System Requirements
- **OS:** Windows 10/11, macOS 10.14+, or Ubuntu 18.04+
- **Python:** 3.8 or higher
- **RAM:** Minimum 8GB (16GB recommended for training)
- **Storage:** At least 5GB free space
- **GPU:** Optional but recommended (CUDA-compatible)

### Required Python Packages
The system will automatically install required packages from `requirements.txt`:

```bash
# Install dependencies
pip install -r requirements.txt
```

**Key Dependencies:**
- `torch>=2.0.0` - Deep learning framework
- `transformers>=4.30.0` - Hugging Face transformers
- `flask>=2.3.0` - Web API framework  
- `pandas>=2.0.0` - Data manipulation
- `scikit-learn>=1.3.0` - Machine learning utilities
- `langdetect>=1.0.9` - Language detection
- `nltk>=3.8.0` - Natural language processing

## 🛠️ Installation Guide

### Step 1: Environment Setup
```bash
# Clone or extract the project
cd C:\fakee

# Create virtual environment (recommended)
python -m venv fake_news_env

# Activate virtual environment
# Windows:
fake_news_env\Scripts\activate
# macOS/Linux:
source fake_news_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Download Required NLTK Data
```python
# Run once to download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Step 3: Verify Installation
```bash
# Run system check
python main.py --mode demo
```

## 🎮 Usage Modes

### 1. Demo Mode (Default)
Demonstrates the system with multilingual sample texts:
```bash
python main.py --mode demo
```

### 2. Web Interface Mode
Starts the Flask web server with a user-friendly interface:
```bash
python main.py --mode api --port 5000
```
Then open: http://localhost:5000

### 3. Training Mode
Train the model with your own dataset:
```bash
python main.py --mode train --data path/to/dataset.csv --epochs 3 --batch_size 16
```

**Dataset Format:**
```csv
text,label
"ಇದು ನಿಜವಾದ ಸುದ್ದಿ",0
"ಈ ಸುದ್ದಿ ಸುಳ್ಳು",1
```
- `text`: News article text
- `label`: 0 for real news, 1 for fake news

### 4. Prediction Mode
Make predictions on individual texts:
```bash
python main.py --mode predict --text "Your news text here"
```

## 🔧 Configuration Options

### Model Selection
Choose between transformer models:
```bash
# Use mBERT (default)
python main.py --model mbert

# Use XLM-RoBERTa
python main.py --model xlm_roberta
```

### Training Parameters
```bash
python main.py --mode train \
    --data dataset.csv \
    --epochs 5 \
    --batch_size 32 \
    --model xlm_roberta
```

### API Configuration
```bash
# Custom port
python main.py --mode api --port 8080

# Set environment variables
export SECRET_KEY="your-secret-key"
export DEBUG=False
```

## 🌐 API Documentation

### REST API Endpoints

#### 1. Web Interface
- **GET /** - Main web interface

#### 2. Prediction API
- **POST /api/predict**
  ```json
  {
    "text": "ಇದು ಸುದ್ದಿ ಪಠ್ಯ",
    "include_features": true
  }
  ```

#### 3. Batch Prediction
- **POST /api/batch_predict**
  ```json
  {
    "texts": ["text1", "text2", "text3"]
  }
  ```

#### 4. System Status
- **GET /api/health** - Health check
- **GET /api/stats** - Performance statistics  
- **GET /api/model_info** - Model information

### API Response Format
```json
{
  "success": true,
  "original_text": "ಇದು ಸುದ್ದಿ ಪಠ್ಯ",
  "cleaned_text": "ಇದು ಸುದ್ದಿ ಪಠ್ಯ",
  "detected_language": "kn",
  "language_name": "Kannada", 
  "prediction": 0,
  "confidence": 0.95,
  "prediction_label": "Real",
  "processing_time": 0.123,
  "features": {...}
}
```

## 📊 Performance Monitoring

### System Metrics
The system tracks performance according to project requirements:

- **Response Time:** <2 seconds per article (target)
- **Accuracy:** 90%+ target on benchmark datasets
- **Throughput:** Thousands of articles per hour capability
- **Language Support:** Kannada, English, Hindi

### View Performance Stats
```bash
# API endpoint
curl http://localhost:5000/api/stats

# Web interface shows real-time stats
```

## 🧪 Testing the System

### 1. Language Detection Test
```python
# Test with different languages
test_texts = [
    "ಇದು ಕನ್ನಡ ಪಠ್ಯ",  # Kannada
    "This is English text",  # English  
    "यह हिंदी पाठ है"  # Hindi
]
```

### 2. Fake News Detection Test
```python
# Test with suspicious patterns
fake_indicators = [
    "!!! ನಂಬಲಾಗದ ಸಂಗತಿ !!!",  # Excessive punctuation
    "BREAKING: Shocking discovery!!!",  # All caps + exclamation
    "यह पूरी तरह झूठ है!!!"  # Hindi fake news pattern
]
```

### 3. Performance Test
```bash
# Load test with multiple requests
python -c "
import requests
import time
for i in range(100):
    start = time.time()
    r = requests.post('http://localhost:5000/api/predict', 
                     json={'text': 'test text'})
    print(f'Request {i}: {time.time()-start:.3f}s')
"
```

## 🚧 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# If you get import errors
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# or
set PYTHONPATH=%PYTHONPATH%;%cd%
```

#### 2. NLTK Data Missing
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

#### 3. Model Download Issues
```bash
# If transformer models don't download
pip install --upgrade transformers
# Clear cache if needed
rm -rf ~/.cache/huggingface/
```

#### 4. Memory Issues
```python
# Reduce batch size
python main.py --mode train --batch_size 8

# Use CPU instead of GPU
export CUDA_VISIBLE_DEVICES=""
```

#### 5. Port Already in Use
```bash
# Use different port
python main.py --mode api --port 8080

# Find and kill process using port 5000
# Windows: netstat -ano | findstr :5000
# Linux/Mac: lsof -ti:5000 | xargs kill -9
```

### Performance Optimization

#### 1. Model Optimization
```python
# Use smaller model for faster inference
python main.py --model mbert  # Instead of xlm_roberta

# Reduce max_length for shorter texts
# Edit config.py: max_length = 256
```

#### 2. System Optimization
```bash
# Enable GPU if available
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Use threading for API
export FLASK_ENV=production
```

## 📁 Project Structure

```
C:\fakee\
├── main.py                    # Main application entry point
├── config.py                  # Configuration settings
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── DEPLOYMENT_GUIDE.md        # This deployment guide
├── fake_news_detection_kannada/  # Additional project files
├── src/                       # Source code
│   ├── api/
│   │   └── fake_news_api.py   # Flask API and web interface
│   ├── data/
│   │   └── multilingual_preprocessor.py  # Text preprocessing
│   └── models/
│       ├── multilingual_feature_extractor.py  # Feature extraction
│       └── multilingual_classifier.py         # Classification model
├── data/                      # Dataset files
├── models/                    # Trained model files
├── results/                   # Evaluation results
└── logs/                      # Application logs
```

## 🔐 Security Considerations

### Data Protection
- Input sanitization implemented
- No sensitive data logging
- Secure API endpoints
- Rate limiting for API calls

### Production Deployment
```bash
# Set production environment variables
export SECRET_KEY="your-secure-secret-key"
export DEBUG=False
export FLASK_ENV=production

# Use WSGI server for production
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 src.api.fake_news_api:app
```

## 📈 Extending the System

### Adding New Languages
1. Update language codes in `config.py`
2. Add stopwords in `multilingual_preprocessor.py`
3. Update Unicode ranges for character detection
4. Retrain models with new language data

### Custom Models
1. Implement new model class inheriting from base
2. Update model configurations
3. Add to main application options

### Additional Features
- Social media integration
- Browser extension
- Mobile app API
- Batch processing capabilities

## 📞 Support & Contact

### Project Team
- **MUHAMMED AQUIF** (4SF22CS114) - Lead Developer
- **PRATHAM AMIN** (4SF22CS145) - NLP Specialist
- **VISHWA** (4SF22CS247) - System Architecture
- **B M SHASHANK** (4SF23CS402) - API Development

### Academic Guide
- **Mrs. SUKETHA** - Assistant Professor, Dept of CSE, SCEM

### Institution
- **Sahyadri College of Engineering & Management**
- **Department of Computer Science & Engineering**
- **Mangaluru, Karnataka, India**

## 📜 License & Citation

This project is developed as part of the Computer Science & Engineering curriculum at Sahyadri College of Engineering & Management.

**Citation:**
```
Multilingual Fake News Detection Using Natural Language Processing for Kannada Language
Authors: Aquif, M., Amin, P., Vishwa, Shashank, B.M.
Guide: Suketha, Mrs.
Institution: Sahyadri College of Engineering & Management, 2024
```

## 🎉 Conclusion

This system successfully implements a multilingual fake news detection solution as specified in the project report, supporting Kannada language along with English and Hindi. The system meets the performance requirements with transformer-based models, real-time processing, and comprehensive evaluation metrics.

For any issues or questions, please refer to the troubleshooting section or contact the development team.

---

**Happy Fake News Detection! 🕵️‍♂️🔍**