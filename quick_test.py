#!/usr/bin/env python3
"""
Quick test script to verify improved fake news detection
"""

import requests
import json

def quick_test():
    api_url = "http://localhost:5000/api/predict"
    
    # Test cases that should be detected as fake
    fake_tests = [
        "BREAKING!!! SHOCKING discovery!!! Scientists dont want you to know this INCREDIBLE secret!!! Click NOW!!!",
        "GET RICH QUICK!!! This SECRET method will make you MILLIONAIRE overnight!!! GUARANTEED!!!",
        "ಈ ಸುದ್ದಿ ಸಂಪೂರ್ಣವಾಗಿ ಸುಳ್ಳು!!! ನಂಬಬೇಡಿ!!! ಇದು ಅಪಾಯಕಾರಿ ಮೋಸ!!!",
        "यह पूरी तरह झूठी खबर है!!! तुरंत सावधान रहें!!! भयानक साजिश!!!"
    ]
    
    # Test cases that should be detected as real
    real_tests = [
        "The government announced new policy changes today. According to official research, implementation will begin next month.",
        "Local authorities report increased traffic in downtown area.",
        "ಸರ್ಕಾರವು ಹೊಸ ನೀತಿಯನ್ನು ಘೋಷಿಸಿದೆ. ಅಧಿಕೃತ ವರದಿ ಪ್ರಕಾರ ಇದು ಮುಂದಿನ ತಿಂಗಳು ಜಾರಿಯಾಗಲಿದೆ."
    ]
    
    print("🔥 QUICK TEST: Improved Fake News Detection")
    print("=" * 50)
    
    print("\n📰 Testing FAKE news samples:")
    fake_correct = 0
    for i, text in enumerate(fake_tests, 1):
        try:
            response = requests.post(api_url, json={"text": text}, timeout=5)
            if response.status_code == 200:
                result = response.json()
                prediction = result["prediction_label"]
                confidence = result["confidence"]
                
                status = "✅ CORRECT" if prediction == "Fake" else "❌ WRONG"
                if prediction == "Fake":
                    fake_correct += 1
                    
                print(f"{i}. {status} - Predicted: {prediction} ({confidence*100:.1f}%)")
                print(f"   Text: {text[:60]}{'...' if len(text) > 60 else ''}")
            else:
                print(f"{i}. ❌ API Error: {response.status_code}")
        except Exception as e:
            print(f"{i}. ❌ Error: {e}")
    
    print(f"\n✅ Fake Detection Accuracy: {fake_correct}/{len(fake_tests)} ({fake_correct/len(fake_tests)*100:.1f}%)")
    
    print("\n📄 Testing REAL news samples:")
    real_correct = 0
    for i, text in enumerate(real_tests, 1):
        try:
            response = requests.post(api_url, json={"text": text}, timeout=5)
            if response.status_code == 200:
                result = response.json()
                prediction = result["prediction_label"]
                confidence = result["confidence"]
                
                status = "✅ CORRECT" if prediction == "Real" else "❌ WRONG"
                if prediction == "Real":
                    real_correct += 1
                    
                print(f"{i}. {status} - Predicted: {prediction} ({confidence*100:.1f}%)")
                print(f"   Text: {text[:60]}{'...' if len(text) > 60 else ''}")
            else:
                print(f"{i}. ❌ API Error: {response.status_code}")
        except Exception as e:
            print(f"{i}. ❌ Error: {e}")
    
    print(f"\n✅ Real Detection Accuracy: {real_correct}/{len(real_tests)} ({real_correct/len(real_tests)*100:.1f}%)")
    
    overall_accuracy = (fake_correct + real_correct) / (len(fake_tests) + len(real_tests)) * 100
    print(f"\n🎯 Overall Accuracy: {overall_accuracy:.1f}%")
    
    if overall_accuracy >= 80:
        print("🎉 EXCELLENT! Model is working well!")
    elif overall_accuracy >= 60:
        print("👍 GOOD! Model improvements are effective!")
    else:
        print("⚠️ Model still needs more tuning.")

if __name__ == "__main__":
    try:
        quick_test()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("💡 Make sure the API server is running at http://localhost:5000")