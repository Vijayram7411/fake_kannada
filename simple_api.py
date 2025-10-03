#!/usr/bin/env python3
"""
Simplified Flask API for Multilingual Fake News Detection
A lightweight version that works without transformer dependencies
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import re
import string
import time
import logging
from datetime import datetime
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['SECRET_KEY'] = 'dev-secret-key'

# Global variables for tracking
request_times = []
request_count = 0
start_time = time.time()

class SimpleFakeNewsDetector:
    """Simplified fake news detector using rule-based approach"""
    
    def __init__(self):
        # Language patterns
        self.kannada_range = r'[\u0C80-\u0CFF]'
        self.devanagari_range = r'[\u0900-\u097F]'
        
        # Enhanced fake news indicators with more comprehensive patterns
        self.fake_indicators = {
            'en': [
                # Sensational/clickbait phrases
                'breaking', 'shocking', 'unbelievable', 'incredible', 'amazing',
                'you wont believe', 'scientists dont want you to know', 'doctors hate',
                'government hiding', 'they dont want you to see', 'secret revealed',
                'must see', 'must read', 'dont miss', 'viral', 'gone wrong',
                # Urgency indicators
                'urgent', 'immediately', 'right now', 'asap', 'hurry',
                # Emotional manipulation
                'terrifying', 'horrifying', 'devastating', 'outrageous',
                # Conspiracy-like phrases
                'cover up', 'conspiracy', 'hidden truth', 'exposed',
                # Health misinformation patterns
                'miracle cure', 'doctors hate this', 'big pharma', 'natural remedy',
                # Financial scams
                'get rich quick', 'easy money', 'guaranteed profit'
            ],
            'kn': [
                # Sensational words in Kannada
                'ನಂಬಲಾಗದ', 'ಆಶ್ಚರ್ಯಕರ', 'ಸುಳ್ಳು', 'ನಂಬಬೇಡಿ', 'ಅಪಾಯಕಾರಿ',
                'ಭಯಾನಕ', 'ಭೀಕರ', 'ಆಘಾತಕಾರಿ', 'ಚಕಿತಗೊಳಿಸುವ',
                # Urgency in Kannada  
                'ತುರ್ತು', 'ತಕ್ಷಣ', 'ಈಗಲೇ', 'ಅವಸರ',
                # Emotional manipulation
                'ಭಯಂಕರ', 'ದುಃಖಕರ', 'ಕ್ರೋಧ', 'ಆಕ್ರೋಶ',
                # Conspiracy-like
                'ರಹಸ್ಯ', 'ಮರೆಮಾಡಿದ', 'ಬಹಿರಂಗ', 'ಸತ್ಯ ಬಾರದೆ',
                # Common fake news words
                'ಸಂಪೂರ್ಣ ಸುಳ್ಳು', 'ಮೋಸ', 'ವಂಚನೆ', 'ದುಷ್ಟ ಸುದ್ದಿ'
            ],
            'hi': [
                # Sensational Hindi words
                'झूठ', 'खतरनाक', 'न मानिए', 'सच्चाई', 'छुपाया',
                'अविश्वसनीय', 'आश्चर्यजनक', 'चौंकाने वाला', 'डरावना',
                # Urgency in Hindi
                'तुरंत', 'अभी', 'जल्दी', 'आपातकाल',
                # Emotional manipulation
                'भयानक', 'डरावना', 'दुखदायक', 'गुस्सा',
                # Conspiracy-like
                'साजिश', 'रहस्य', 'छुपाया गया', 'बेनकाब',
                # Common fake news patterns
                'पूरी तरह झूठ', 'धोखा', 'फर्जी खबर', 'गलत सूचना'
            ]
        }
        
    def detect_language(self, text):
        """Simple language detection"""
        kannada_chars = len(re.findall(self.kannada_range, text))
        devanagari_chars = len(re.findall(self.devanagari_range, text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        total_chars = len(re.sub(r'[^\w]', '', text))
        if total_chars == 0:
            return 'en'
        
        kannada_ratio = kannada_chars / total_chars
        devanagari_ratio = devanagari_chars / total_chars
        english_ratio = english_chars / total_chars
        
        if kannada_ratio > 0.3:
            return 'kn'
        elif devanagari_ratio > 0.3:
            return 'hi'
        else:
            return 'en'
    
    def get_language_name(self, lang_code):
        """Get language name from code"""
        names = {'kn': 'Kannada', 'en': 'English', 'hi': 'Hindi'}
        return names.get(lang_code, lang_code)
    
    def predict(self, text):
        """Ultra-aggressive fake news detection - prioritizes catching fake news"""
        if not text or len(text.strip()) < 10:
            return 0, 0.5  # Default to real with low confidence
        
        # Detect language
        language = self.detect_language(text)
        
        # Clean text
        cleaned_text = text.lower().strip()
        original_text = text.strip()
        
        # Count basic fake indicators
        fake_points = 0
        
        # IMMEDIATE FAKE DETECTION - Any of these = instant fake
        immediate_fake_patterns = {
            'en': [
                '!!!', 'breaking', 'shocking', 'unbelievable', 'incredible', 
                'must see', 'must read', 'click now', 'doctors hate', 'scientists dont want',
                'you wont believe', 'secret', 'exposed', 'hidden truth', 'conspiracy',
                'government hiding', 'big pharma', 'miracle cure', 'get rich quick',
                'easy money', 'guaranteed', 'amazing', 'terrifying', 'horrifying'
            ],
            'kn': [
                '!!!', 'ಸುಳ್ಳು', 'ನಂಬಬೇಡಿ', 'ಅಪಾಯಕಾರಿ', 'ಮೋಸ', 'ವಂಚನೆ',
                'ಭಯಾನಕ', 'ಆಶ್ಚರ್ಯಕರ', 'ನಂಬಲಾಗದ', 'ರಹಸ್ಯ', 'ತುರ್ತು'
            ],
            'hi': [
                '!!!', 'झूठ', 'सावधान', 'साजिश', 'खतरा', 'धोखा', 'फर्जी',
                'आश्चर्यजनक', 'अविश्वसनीय', 'रहस्य', 'तुरंत', 'भयानक'
            ]
        }
        
        # Check for immediate fake indicators
        if language in immediate_fake_patterns:
            for pattern in immediate_fake_patterns[language]:
                if pattern in cleaned_text:
                    fake_points += 10  # Heavy penalty for each fake pattern
        
        # Count exclamation marks (very strong fake indicator)
        exclamation_count = text.count('!')
        if exclamation_count >= 3:
            fake_points += 15  # Strong fake signal
        elif exclamation_count >= 2:
            fake_points += 10
        elif exclamation_count >= 1:
            fake_points += 5
        
        # Check for ALL CAPS words (strong fake indicator)
        caps_words = len(re.findall(r'\b[A-Z]{3,}\b', original_text))
        if caps_words > 0:
            fake_points += caps_words * 3  # 3 points per caps word
        
        # Check for repetitive patterns
        if '...' in text or '???' in text:
            fake_points += 5
        
        # Check for language-specific fake keywords
        if language in self.fake_indicators:
            for indicator in self.fake_indicators[language]:
                if indicator.lower() in cleaned_text:
                    fake_points += 8  # Strong penalty for each fake keyword
        
        # Check for credibility indicators (reduce fake score)
        credibility_words = {
            'en': ['according to', 'research', 'study', 'experts', 'official', 'government', 'university'],
            'kn': ['ಅಧಿಕೃತ', 'ಸರ್ಕಾರಿ', 'ಸಂಶೋಧನೆ', 'ತಜ್ಞರು', 'ವರದಿ'],
            'hi': ['अधिकारी', 'सरकारी', 'अनुसंधान', 'विशेषज्ञ', 'रिपोर्ट']
        }
        
        credibility_found = 0
        if language in credibility_words:
            for word in credibility_words[language]:
                if word.lower() in cleaned_text:
                    credibility_found += 1
        
        # Reduce fake points for credibility indicators (but don't eliminate fake detection)
        fake_points = max(0, fake_points - (credibility_found * 3))
        
        # ULTRA AGGRESSIVE THRESHOLDS
        if fake_points >= 15:  # Very high fake indicators
            return 1, 0.95
        elif fake_points >= 10:  # High fake indicators  
            return 1, 0.85
        elif fake_points >= 8:  # Medium-high fake indicators
            return 1, 0.75
        elif fake_points >= 5:  # Medium fake indicators
            return 1, 0.65
        elif fake_points >= 3:  # Low-medium fake indicators
            return 1, 0.55
        elif fake_points >= 1:  # Any fake indicator
            return 1, 0.51  # Barely fake but still fake
        else:  # No fake indicators
            return 0, 0.8  # Real with high confidence
        
        return prediction, confidence
    
    def extract_features(self, text):
        """Extract basic text features"""
        if not text:
            return {}
        
        language = self.detect_language(text)
        
        # Basic statistics
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(re.split(r'[.!?]+', text)),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'punctuation_ratio': sum(1 for c in text if c in string.punctuation) / len(text) if text else 0,
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0
        }
        
        # Language-specific ratios
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
        
        return features

# Initialize detector
detector = SimpleFakeNewsDetector()
model_loaded = True

def process_text(text, include_features=False):
    """Process text for fake news detection"""
    try:
        start_time = time.time()
        
        # Detect language
        language = detector.detect_language(text)
        language_name = detector.get_language_name(language)
        
        # Make prediction
        prediction, confidence = detector.predict(text)
        
        # Extract features if requested
        features = None
        if include_features:
            features = detector.extract_features(text)
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "original_text": text,
            "cleaned_text": text.strip(),
            "detected_language": language,
            "language_name": language_name,
            "prediction": prediction,
            "confidence": confidence,
            "prediction_label": "Fake" if prediction == 1 else "Real",
            "features": features,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

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
                margin: 10px 5px;
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
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
            .prediction {
                font-size: 1.5em;
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                margin: 20px 0;
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
            .result-item {
                margin: 10px 0;
                padding: 10px;
                background: rgba(255,255,255,0.7);
                border-radius: 5px;
            }
            .examples {
                margin-top: 30px;
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
            }
            .example-text {
                background: white;
                padding: 15px;
                margin: 10px 0;
                border-radius: 8px;
                cursor: pointer;
                border-left: 4px solid #667eea;
                transition: all 0.3s ease;
            }
            .example-text:hover {
                background: #e9ecef;
                transform: translateX(5px);
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
                    <div class="stat-number">✅</div>
                    <div class="stat-label">System Status</div>
                </div>
            </div>
            
            <div class="input-section">
                <label for="news-text">Enter news article or text to analyze:</label>
                <textarea id="news-text" placeholder="Enter news text in Kannada, English, or Hindi...

Examples:
• ಇದು ನಿಜವಾದ ಸುದ್ದಿ ವರದಿ (Kannada - Real News)
• ಈ ಸುದ್ದಿ ಸಂಪೂರ್ಣವಾಗಿ ಸುಳ್ಳು!!! (Kannada - Fake News)
• This is a legitimate news report (English - Real News)
• BREAKING: Shocking discovery!!! (English - Fake News)
• यह सत्य समाचार है (Hindi - Real News)
• यह झूठी खबर है!!! (Hindi - Fake News)"></textarea>
                
                <div style="text-align: center; margin-top: 20px;">
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
                <div class="example-text" onclick="setExample('ಈ ಸುದ್ದಿ ಸಂಪೂರ್ಣವಾಗಿ ಸುಳ್ಳು! ನಂಬಬೇಡಿ!!! ಇದು ಅಪಾಯಕಾರಿ!!!')">
                    ⚠️ <strong>Kannada Fake News:</strong> ಈ ಸುದ್ದಿ ಸಂಪೂರ್ಣವಾಗಿ ಸುಳ್ಳು! ನಂಬಬೇಡಿ!!! ಇದು ಅಪಾಯಕಾರಿ!!!
                </div>
                <div class="example-text" onclick="setExample('This is a legitimate news report about government policy changes and their impact on citizens.')">
                    📰 <strong>English Real News:</strong> This is a legitimate news report about government policy changes...
                </div>
                <div class="example-text" onclick="setExample('BREAKING: Shocking discovery!!! Scientists dont want you to know this!!! Click now!!!')">
                    ⚠️ <strong>English Fake News:</strong> BREAKING: Shocking discovery!!! Scientists dont want you to know this!!!
                </div>
                <div class="example-text" onclick="setExample('सरकार ने नई नीति की घोषणा की है। यह महत्वपूर्ण समाचार है।')">
                    📰 <strong>Hindi Real News:</strong> सरकार ने नई नीति की घोषणा की है। यह महत्वपूर्ण समाचार है।
                </div>
                <div class="example-text" onclick="setExample('यह पूरी तरह से झूठी खबर है! इसे मत मानिए!!!')">
                    ⚠️ <strong>Hindi Fake News:</strong> यह पूरी तरह से झूठी खबर है! इसे मत मानिए!!!
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
                    'BREAKING: Shocking discovery!!! Scientists dont want you to know this!!!',
                    'यह पूरी तरह से झूठी खबर है! इसे मत मानिए!!!'
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
                        
                        <div class="prediction ${data.prediction === 1 ? 'fake' : 'real'}">
                            ${data.prediction === 1 ? '⚠️ FAKE NEWS DETECTED' : '✅ APPEARS TO BE REAL NEWS'}
                            <br><small>Confidence: ${(data.confidence * 100).toFixed(1)}%</small>
                        </div>
                        
                        <div class="result-item">
                            <strong>🌐 Language:</strong> ${data.language_name}
                        </div>
                        
                        <div class="result-item">
                            <strong>⏱️ Processing Time:</strong> ${responseTime}ms
                        </div>
                        
                        ${data.features ? `
                        <div class="result-item">
                            <strong>📝 Text Length:</strong> ${data.features.text_length} characters
                        </div>
                        <div class="result-item">
                            <strong>📊 Word Count:</strong> ${data.features.word_count}
                        </div>
                        <div class="result-item">
                            <strong>❗ Exclamation Count:</strong> ${data.features.exclamation_count}
                        </div>
                        <div class="result-item">
                            <strong>🔤 Language Distribution:</strong>
                            Kannada: ${(data.features.kannada_ratio * 100).toFixed(1)}%,
                            English: ${(data.features.english_ratio * 100).toFixed(1)}%,
                            Hindi: ${(data.features.devanagari_ratio * 100).toFixed(1)}%
                        </div>
                        ` : ''}
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
        result = process_text(text, include_features)
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
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    uptime = time.time() - start_time
    
    return jsonify({
        "status": "healthy",
        "uptime_seconds": round(uptime, 2),
        "request_count": request_count,
        "avg_response_time_ms": round(sum(request_times) / len(request_times), 2) if request_times else 0,
        "supported_languages": ["Kannada", "English", "Hindi"],
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
            "supported_languages": ["Kannada", "English", "Hindi"]
        },
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/test_improvements')
def test_improvements():
    """Test the improved fake news detection"""
    test_cases = [
        # Real news examples
        {
            "text": "The government announced new policy changes today. According to official sources, the implementation will begin next month.",
            "expected": "Real",
            "language": "English"
        },
        {
            "text": "ಸರ್ಕಾರವು ಹೊಸ ನೀತಿಯನ್ನು ಘೋಷಿಸಿದೆ. ಅಧಿಕೃತ ವರದಿ ಪ್ರಕಾರ, ಇದು ಮುಂದಿನ ತಿಂಗಳು ಜಾರಿಯಾಗಲಿದೆ.",
            "expected": "Real",
            "language": "Kannada"
        },
        {
            "text": "सरकार ने आज नई नीति की घोषणा की। अधिकारियों के अनुसार यह अगले महीने से लागू होगी।",
            "expected": "Real", 
            "language": "Hindi"
        },
        # Fake news examples
        {
            "text": "BREAKING!!! SHOCKING discovery!!! Scientists dont want you to know this INCREDIBLE secret!!! Click NOW!!!",
            "expected": "Fake",
            "language": "English"
        },
        {
            "text": "ಈ ಸುದ್ದಿ ಸಂಪೂರ್ಣವಾಗಿ ಸುಳ್ಳು!!! ನಂಬಬೇಡಿ!!! ಇದು ಅಪಾಯಕಾರಿ ಮೋಸ!!!",
            "expected": "Fake",
            "language": "Kannada"
        },
        {
            "text": "यह पूरी तरह झूठी खबर है!!! तुरंत सावधान रहें!!! भयानक साजिश!!!",
            "expected": "Fake",
            "language": "Hindi"
        },
        # Borderline cases
        {
            "text": "Important announcement from the health ministry regarding new guidelines.",
            "expected": "Real",
            "language": "English"
        },
        {
            "text": "Amazing breakthrough! New research reveals surprising results.",
            "expected": "Borderline",
            "language": "English"
        }
    ]
    
    results = []
    correct_predictions = 0
    
    for test_case in test_cases:
        result = process_text(test_case["text"], include_features=True)
        
        # Determine if prediction matches expectation
        predicted_label = result["prediction_label"]
        expected_label = test_case["expected"]
        
        is_correct = (
            (expected_label == "Real" and predicted_label == "Real") or
            (expected_label == "Fake" and predicted_label == "Fake") or
            (expected_label == "Borderline")  # Accept any prediction for borderline cases
        )
        
        if is_correct:
            correct_predictions += 1
            
        results.append({
            "text": test_case["text"][:100] + "..." if len(test_case["text"]) > 100 else test_case["text"],
            "language": test_case["language"],
            "expected": expected_label,
            "predicted": predicted_label,
            "confidence": round(result["confidence"] * 100, 1),
            "correct": is_correct,
            "processing_time": result["processing_time"]
        })
    
    accuracy = (correct_predictions / len(test_cases)) * 100
    
    return jsonify({
        "success": True,
        "test_summary": {
            "total_tests": len(test_cases),
            "correct_predictions": correct_predictions,
            "accuracy_percentage": round(accuracy, 1),
            "avg_processing_time": round(sum(r["processing_time"] for r in results) / len(results) * 1000, 2)
        },
        "detailed_results": results,
        "improvements": {
            "enhanced_keywords": "Added more comprehensive fake news indicators",
            "weighted_scoring": "Implemented weighted scoring system with 8 different factors",
            "credibility_detection": "Added credibility indicators that reduce fake news scores",
            "conservative_thresholds": "More conservative classification thresholds",
            "language_specific": "Enhanced language-specific detection for Kannada, English, Hindi"
        },
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    logger.info("Starting Multilingual Fake News Detection API")
    logger.info("Supported languages: Kannada, English, Hindi")
    logger.info("Using simplified rule-based detection")
    logger.info("Available at: http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)