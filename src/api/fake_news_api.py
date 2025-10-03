"""
Flask API for Multilingual Fake News Detection
Provides REST API endpoints and web interface for real-time classification
"""

from flask import Flask, request, jsonify, render_template_string, send_from_directory
from flask_cors import CORS
import os
import sys
import time
import logging
import json
from typing import Dict, List, Optional
import traceback
import hashlib
from datetime import datetime
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import joblib

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import our modules
from src.models.multilingual_classifier import MultilingualFakeNewsDetector
from src.data.multilingual_preprocessor import MultilingualPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max request size

# Global variables for model and preprocessor
detector = None
preprocessor = None
model_loaded = False
loading_lock = threading.Lock()

# Fallback TF-IDF + Logistic Regression (as per report’s best model)
tfidf_vectorizer = None
lr_model = None

# Performance monitoring
request_times = []
request_count = 0
start_time = time.time()

# Thread pool for concurrent processing
executor = ThreadPoolExecutor(max_workers=4)

def load_models():
    """Load the trained models and preprocessor"""
    global detector, preprocessor, model_loaded, tfidf_vectorizer, lr_model
    
    with loading_lock:
        if model_loaded:
            return
        
        try:
            logger.info("Loading multilingual fake news detector...")
            
            # Initialize preprocessor
            preprocessor = MultilingualPreprocessor()
            logger.info("Preprocessor loaded successfully")
            
            # Initialize detector (will load model if available)
            detector = MultilingualFakeNewsDetector(
                model_name='bert-base-multilingual-cased',
                max_length=512
            )
            
            # Try to load pre-trained transformer model if available
            model_path = os.path.join('models', 'multilingual_fake_news_model.pth')
            if os.path.exists(model_path):
                try:
                    detector.load_model(model_path)
                    logger.info(f"Pre-trained transformer model loaded from {model_path}")
                except Exception as e:
                    logger.warning(f"Could not load pre-trained transformer model: {e}")
                    logger.info("Proceeding without transformer model; will use fallback if available")
            else:
                logger.info("No pre-trained transformer model found; will use fallback if available")
            
            # Load TF-IDF + Logistic Regression fallback (recommended by report)
            try:
                vec_path = os.path.join('models', 'tfidf_vectorizer.joblib')
                lr_path = os.path.join('models', 'logistic_regression.joblib')
                if os.path.exists(vec_path) and os.path.exists(lr_path):
                    tfidf_vectorizer = joblib.load(vec_path)
                    lr_model = joblib.load(lr_path)
                    logger.info("TF-IDF + Logistic Regression fallback loaded successfully")
                else:
                    logger.info("TF-IDF fallback artifacts not found; API will run without fallback")
            except Exception as e:
                logger.warning(f"Failed to load TF-IDF fallback models: {e}")
            
            model_loaded = True
            logger.info("Models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            logger.error(traceback.format_exc())
            raise

def get_model_info() -> Dict:
    """Get information about loaded models"""
    if not model_loaded:
        return {"status": "not_loaded", "message": "Models not loaded"}
    
    return {
        "status": "loaded",
        "detector_model": detector.model_name if detector else "Not available",
        "max_length": detector.max_length if detector else "Not available",
        "supported_languages": ["Kannada", "English", "Hindi"],
        "model_loaded": model_loaded,
        "fallback_available": bool(tfidf_vectorizer is not None and lr_model is not None),
        "fallback_model": "TF-IDF + Logistic Regression" if (tfidf_vectorizer and lr_model) else None
    }

def process_text_async(text: str, include_features: bool = False) -> Dict:
    """Process text asynchronously for better performance"""
    try:
        start_time = time.time()
        
        # Preprocess text
        language = preprocessor.detect_language(text)
        cleaned_text = preprocessor.clean_text(text, language)
        
        # Extract features if requested
        features = None
        if include_features:
            features = preprocessor.extract_language_features(text)
        
        # Make prediction using transformer model if available, otherwise fallback
        prediction = None
        confidence = None
        
        used_backend = None
        if detector and detector.model is not None:
            try:
                predictions, confidences = detector.predict([cleaned_text])
                prediction = int(predictions[0])
                confidence = float(confidences[0])
                used_backend = 'transformer'
            except Exception as e:
                logger.warning(f"Transformer prediction failed: {e}")
                prediction = None
                confidence = None
        
        # Fallback: TF-IDF + Logistic Regression (per report recommendation)
        if prediction is None and tfidf_vectorizer is not None and lr_model is not None:
            try:
                X = tfidf_vectorizer.transform([cleaned_text])
                pred = int(lr_model.predict(X)[0])
                proba = lr_model.predict_proba(X)[0]
                conf = float(proba[pred]) if hasattr(lr_model, 'predict_proba') else 0.75
                prediction = pred
                confidence = conf
                used_backend = 'tfidf_lr'
            except Exception as e:
                logger.warning(f"TF-IDF fallback prediction failed: {e}")
                prediction = None
                confidence = None
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "original_text": text,
            "cleaned_text": cleaned_text,
            "detected_language": language,
            "language_name": preprocessor.language_names.get(language, language),
            "prediction": prediction,
            "confidence": confidence,
            "prediction_label": "Fake" if prediction == 1 else "Real" if prediction == 0 else "Unknown",
            "features": features,
            "processing_time": processing_time,
            "backend": used_backend,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.before_first_request
def initialize():
    """Initialize models before first request"""
    try:
        load_models()
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")

@app.route('/')
def home():
    """Main web interface"""
    html_template = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Multilingual Fake News Detection - Kannada Language Support</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: #333;
            }
            .container {
                max-width: 1000px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 15px 35px rgba(0,0,0,0.1);
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
                border-bottom: 2px solid #e0e0e0;
                padding-bottom: 20px;
            }
            .header h1 {
                color: #2c3e50;
                margin: 0 0 10px 0;
                font-size: 2.5em;
            }
            .header p {
                color: #7f8c8d;
                font-size: 1.1em;
                margin: 0;
            }
            .lang-support {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 25px;
                text-align: center;
            }
            .lang-badges {
                display: flex;
                justify-content: center;
                gap: 15px;
                margin-top: 10px;
                flex-wrap: wrap;
            }
            .lang-badge {
                background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
                color: white;
                padding: 8px 16px;
                border-radius: 20px;
                font-weight: bold;
                font-size: 0.9em;
            }
            .input-section {
                margin-bottom: 25px;
            }
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 8px;
                font-weight: bold;
                color: #2c3e50;
            }
            textarea {
                width: 100%;
                min-height: 150px;
                padding: 15px;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                font-size: 14px;
                resize: vertical;
                font-family: inherit;
                box-sizing: border-box;
            }
            textarea:focus {
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
            .button-group {
                display: flex;
                gap: 15px;
                justify-content: center;
                flex-wrap: wrap;
            }
            button {
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 25px;
                cursor: pointer;
                font-size: 16px;
                font-weight: bold;
                transition: all 0.3s ease;
                min-width: 120px;
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }
            button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            .results {
                margin-top: 30px;
                padding: 20px;
                border-radius: 10px;
                display: none;
            }
            .results.success {
                background: #d4edda;
                border-left: 5px solid #28a745;
            }
            .results.error {
                background: #f8d7da;
                border-left: 5px solid #dc3545;
            }
            .result-item {
                margin: 10px 0;
                padding: 10px;
                background: rgba(255,255,255,0.7);
                border-radius: 5px;
            }
            .result-label {
                font-weight: bold;
                display: inline-block;
                min-width: 120px;
            }
            .prediction {
                font-size: 1.3em;
                padding: 10px;
                border-radius: 8px;
                text-align: center;
                margin: 15px 0;
                font-weight: bold;
            }
            .prediction.real {
                background: linear-gradient(45deg, #28a745, #20c997);
                color: white;
            }
            .prediction.fake {
                background: linear-gradient(45deg, #dc3545, #fd7e14);
                color: white;
            }
            .loading {
                text-align: center;
                padding: 20px;
                display: none;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto 15px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .examples {
                margin-top: 30px;
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
            }
            .example-text {
                background: white;
                padding: 10px;
                margin: 10px 0;
                border-radius: 5px;
                cursor: pointer;
                border-left: 4px solid #667eea;
                transition: all 0.3s ease;
            }
            .example-text:hover {
                background: #e9ecef;
                transform: translateX(5px);
            }
            .stats {
                display: flex;
                justify-content: space-around;
                margin: 20px 0;
                flex-wrap: wrap;
            }
            .stat-item {
                text-align: center;
                padding: 15px;
                background: #f8f9fa;
                border-radius: 10px;
                margin: 5px;
                min-width: 120px;
            }
            .stat-number {
                font-size: 2em;
                font-weight: bold;
                color: #667eea;
            }
            .stat-label {
                font-size: 0.9em;
                color: #666;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔍 Multilingual Fake News Detection</h1>
                <p>Advanced NLP system for detecting fake news in Kannada, English, and Hindi languages</p>
            </div>
            
            <div class="lang-support">
                <h3>Supported Languages</h3>
                <div class="lang-badges">
                    <span class="lang-badge">ಕನ್ನಡ - Kannada</span>
                    <span class="lang-badge">English</span>
                    <span class="lang-badge">हिंदी - Hindi</span>
                </div>
            </div>
            
            <div class="stats">
                <div class="stat-item">
                    <div class="stat-number" id="total-requests">0</div>
                    <div class="stat-label">Total Requests</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number" id="avg-time">0ms</div>
                    <div class="stat-label">Avg Response Time</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">90%+</div>
                    <div class="stat-label">Target Accuracy</div>
                </div>
            </div>
            
            <div class="input-section">
                <div class="form-group">
                    <label for="news-text">Enter news article, headline, or URL:</label>
                    <textarea id="news-text" placeholder="Enter news text in Kannada, English, or Hindi...

Examples:
ಈ ಸುದ್ದಿ ನಿಜವೇ? (Kannada)
Is this news real? (English)
क्या यह समाचार सत्य है? (Hindi)"></textarea>
                </div>
                
                <div class="button-group">
                    <button onclick="analyzeNews()">🔍 Analyze News</button>
                    <button onclick="clearResults()">🗑️ Clear</button>
                    <button onclick="loadExample()">📰 Load Example</button>
                </div>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyzing news article... Please wait.</p>
            </div>
            
            <div class="results" id="results">
                <!-- Results will be populated here -->
            </div>
            
            <div class="examples">
                <h3>Example Texts (Click to test)</h3>
                <div class="example-text" onclick="setExample('ಇದು ನಿಜವಾದ ಸುದ್ದಿ ವರದಿಯಾಗಿದೆ. ಸರ್ಕಾರವು ಹೊಸ ನೀತಿಯನ್ನು ಜಾರಿಗೊಳಿಸಿದೆ.')">
                    📰 <strong>Kannada Real News:</strong> ಇದು ನಿಜವಾದ ಸುದ್ದಿ ವರದಿಯಾಗಿದೆ. ಸರ್ಕಾರವು ಹೊಸ ನೀತಿಯನ್ನು ಜಾರಿಗೊಳಿಸಿದೆ.
                </div>
                <div class="example-text" onclick="setExample('ಇದು ಸಂಪೂರ್ಣವಾಗಿ ಸುಳ್ಳು ಸುದ್ದಿ! ನಂಬಬೇಡಿ!!! ಇದು ಅಪಾಯಕಾರಿ!!!')">
                    ⚠️ <strong>Kannada Fake News:</strong> ಇದು ಸಂಪೂರ್ಣವಾಗಿ ಸುಳ್ಳು ಸುದ್ದಿ! ನಂಬಬೇಡಿ!!! ಇದು ಅಪಾಯಕಾರಿ!!!
                </div>
                <div class="example-text" onclick="setExample('This is a legitimate news report about government policy changes and their impact on citizens.')">
                    📰 <strong>English Real News:</strong> This is a legitimate news report about government policy changes...
                </div>
                <div class="example-text" onclick="setExample('BREAKING: Shocking discovery!!! Scientists dont want you to know this!!! Click now!!!')">
                    ⚠️ <strong>English Fake News:</strong> BREAKING: Shocking discovery!!! Scientists dont want you to know this!!!
                </div>
            </div>
        </div>
        
        <script>
            let requestCount = 0;
            let totalTime = 0;
            
            function updateStats(responseTime) {
                requestCount++;
                totalTime += responseTime;
                
                document.getElementById('total-requests').textContent = requestCount;
                document.getElementById('avg-time').textContent = Math.round(totalTime / requestCount) + 'ms';
            }
            
            function setExample(text) {
                document.getElementById('news-text').value = text;
            }
            
            function loadExample() {
                const examples = [
                    'ಇದು ನಿಜವಾದ ಸುದ್ದಿ ವರದಿಯಾಗಿದೆ. ಸರ್ಕಾರವು ಹೊಸ ನೀತಿಯನ್ನು ಜಾರಿಗೊಳಿಸಿದೆ.',
                    'This is a legitimate news report about recent policy changes.',
                    'सरकार ने नई नीति की घोषणा की है। यह महत्वपूर्ण समाचार है।',
                    'ಈ ಸುದ್ದಿ ಸಂಪೂರ್ಣವಾಗಿ ಸುಳ್ಳು! ನಂಬಬೇಡಿ!!!',
                ];
                const randomExample = examples[Math.floor(Math.random() * examples.length)];
                setExample(randomExample);
            }
            
            async function analyzeNews() {
                const text = document.getElementById('news-text').value.trim();
                const loadingDiv = document.getElementById('loading');
                const resultsDiv = document.getElementById('results');
                
                if (!text) {
                    alert('Please enter some text to analyze.');
                    return;
                }
                
                // Show loading
                loadingDiv.style.display = 'block';
                resultsDiv.style.display = 'none';
                
                const startTime = Date.now();
                
                try {
                    const response = await fetch('/api/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            text: text,
                            include_features: true
                        })
                    });
                    
                    const responseTime = Date.now() - startTime;
                    updateStats(responseTime);
                    
                    const data = await response.json();
                    
                    loadingDiv.style.display = 'none';
                    displayResults(data, responseTime);
                    
                } catch (error) {
                    loadingDiv.style.display = 'none';
                    displayError('Network error: ' + error.message);
                }
            }
            
            function displayResults(data, responseTime) {
                const resultsDiv = document.getElementById('results');
                
                if (data.success) {
                    resultsDiv.className = 'results success';
                    resultsDiv.innerHTML = `
                        <h3>📊 Analysis Results</h3>
                        
                        ${data.prediction !== null ? `
                        <div class="prediction ${data.prediction === 1 ? 'fake' : 'real'}">
                            ${data.prediction === 1 ? '⚠️ FAKE NEWS DETECTED' : '✅ APPEARS TO BE REAL NEWS'}
                            <br><small>Confidence: ${(data.confidence * 100).toFixed(1)}%</small>
                        </div>
                        ` : '<div class="prediction" style="background: #6c757d;">🤖 Model not trained yet</div>'}
                        
                        <div class="result-item">
                            <span class="result-label">🌐 Language:</span> ${data.language_name}
                        </div>
                        
                        <div class="result-item">
                            <span class="result-label">⏱️ Processing Time:</span> ${responseTime}ms
                        </div>
                        
                        ${data.features ? `
                        <div class="result-item">
                            <span class="result-label">📝 Text Length:</span> ${data.features.text_length} characters
                        </div>
                        <div class="result-item">
                            <span class="result-label">📊 Words:</span> ${data.features.word_count}
                        </div>
                        <div class="result-item">
                            <span class="result-label">🔤 Language Distribution:</span>
                            Kannada: ${(data.features.kannada_ratio * 100).toFixed(1)}%,
                            English: ${(data.features.english_ratio * 100).toFixed(1)}%,
                            Hindi: ${(data.features.devanagari_ratio * 100).toFixed(1)}%
                        </div>
                        ` : ''}
                        
                        <div class="result-item">
                            <span class="result-label">🧹 Cleaned Text:</span>
                            <div style="max-height: 100px; overflow-y: auto; background: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 5px;">
                                ${data.cleaned_text}
                            </div>
                        </div>
                    `;
                } else {
                    displayError(data.error || 'Unknown error occurred');
                }
                
                resultsDiv.style.display = 'block';
            }
            
            function displayError(error) {
                const resultsDiv = document.getElementById('results');
                resultsDiv.className = 'results error';
                resultsDiv.innerHTML = `
                    <h3>❌ Error</h3>
                    <div class="result-item">
                        <strong>Error Message:</strong> ${error}
                    </div>
                `;
                resultsDiv.style.display = 'block';
            }
            
            function clearResults() {
                document.getElementById('news-text').value = '';
                document.getElementById('results').style.display = 'none';
                document.getElementById('loading').style.display = 'none';
            }
            
            // Load initial stats on page load
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('total-requests').textContent = data.stats.total_requests;
                        if (data.stats.total_requests > 0) {
                            document.getElementById('avg-time').textContent = 
                                Math.round(data.stats.avg_response_time) + 'ms';
                        }
                    }
                });
        </script>
    </body>
    </html>
    '''
    return render_template_string(html_template)

@app.route('/api/predict', methods=['POST'])
def predict_news():
    """API endpoint for news prediction"""
    global request_count, request_times
    
    try:
        # Check if models are loaded
        if not model_loaded:
            return jsonify({
                "success": False,
                "error": "Models not loaded. Please wait for initialization to complete.",
                "timestamp": datetime.now().isoformat()
            }), 503
        
        # Get request data
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                "success": False,
                "error": "No text provided in request",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({
                "success": False,
                "error": "Empty text provided",
                "timestamp": datetime.now().isoformat()
            }), 400
        
        include_features = data.get('include_features', False)
        
        # Process text
        start_time = time.time()
        result = process_text_async(text, include_features)
        processing_time = time.time() - start_time
        
        # Update performance tracking
        request_count += 1
        request_times.append(processing_time * 1000)  # Convert to milliseconds
        if len(request_times) > 1000:  # Keep only last 1000 requests
            request_times.pop(0)
        
        # Add request ID for tracking
        result['request_id'] = hashlib.md5(f"{text}{time.time()}".encode()).hexdigest()[:8]
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """API endpoint for batch prediction"""
    try:
        if not model_loaded:
            return jsonify({
                "success": False,
                "error": "Models not loaded"
            }), 503
        
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({
                "success": False,
                "error": "No texts provided"
            }), 400
        
        texts = data['texts']
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({
                "success": False,
                "error": "Invalid texts format"
            }), 400
        
        # Limit batch size
        if len(texts) > 100:
            return jsonify({
                "success": False,
                "error": "Batch size too large (max 100)"
            }), 400
        
        # Process batch
        results = []
        for text in texts:
            result = process_text_async(str(text), False)
            results.append(result)
        
        return jsonify({
            "success": True,
            "results": results,
            "batch_size": len(texts),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch API Error: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    uptime = time.time() - start_time
    
    return jsonify({
        "status": "healthy" if model_loaded else "loading",
        "uptime_seconds": round(uptime, 2),
        "model_info": get_model_info(),
        "request_count": request_count,
        "avg_response_time_ms": round(sum(request_times) / len(request_times), 2) if request_times else 0,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/stats')
def get_stats():
    """Get API statistics"""
    uptime = time.time() - start_time
    
    return jsonify({
        "success": True,
        "stats": {
            "total_requests": request_count,
            "avg_response_time": round(sum(request_times) / len(request_times), 2) if request_times else 0,
            "uptime_seconds": round(uptime, 2),
            "requests_per_second": round(request_count / uptime, 2) if uptime > 0 else 0,
            "model_loaded": model_loaded,
            "supported_languages": ["Kannada", "English", "Hindi"]
        },
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/model_info')
def model_info():
    """Get model information"""
    return jsonify({
        "success": True,
        "model_info": get_model_info(),
        "timestamp": datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint not found",
        "available_endpoints": [
            "GET /",
            "POST /api/predict",
            "POST /api/batch_predict", 
            "GET /api/health",
            "GET /api/stats",
            "GET /api/model_info"
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error",
        "timestamp": datetime.now().isoformat()
    }), 500

if __name__ == '__main__':
    # Initialize models
    try:
        load_models()
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        logger.info("Starting server without models loaded. They will be loaded on first request.")
    
    # Start the Flask application
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Multilingual Fake News Detection API on port {port}")
    logger.info("Supported languages: Kannada, English, Hindi")
    logger.info("Available endpoints:")
    logger.info("  GET  /                 - Web interface")
    logger.info("  POST /api/predict      - Single text prediction")
    logger.info("  POST /api/batch_predict - Batch prediction")
    logger.info("  GET  /api/health       - Health check")
    logger.info("  GET  /api/stats        - API statistics")
    logger.info("  GET  /api/model_info   - Model information")
    
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)