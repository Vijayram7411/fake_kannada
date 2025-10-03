#!/usr/bin/env python3
"""
Ultra-quick test to verify extreme fake news detection sensitivity
"""

import requests

def ultra_test():
    api_url = "http://localhost:5000/api/predict"
    
    print("🚨 ULTRA-AGGRESSIVE FAKE NEWS DETECTION TEST")
    print("=" * 55)
    
    # These should ALL be detected as FAKE now
    tests = [
        {
            "text": "BREAKING!!! SHOCKING discovery!!! Scientists dont want you to know this INCREDIBLE secret!!! Click NOW!!!",
            "should_be": "Fake",
            "description": "Multiple exclamations + clickbait"
        },
        {
            "text": "GET RICH QUICK!!! This SECRET method will make you MILLIONAIRE overnight!!! GUARANTEED!!!",
            "should_be": "Fake", 
            "description": "Financial scam language"
        },
        {
            "text": "ಈ ಸುದ್ದಿ ಸಂಪೂರ್ಣವಾಗಿ ಸುಳ್ಳು!!! ನಂಬಬೇಡಿ!!! ಇದು ಅಪಾಯಕಾರಿ ಮೋಸ!!!",
            "should_be": "Fake",
            "description": "Kannada explicit fake indicators"
        },
        {
            "text": "यह पूरी तरह झूठी खबर है!!! तुरंत सावधान रहें!!! भयानक साजिश!!!",
            "should_be": "Fake",
            "description": "Hindi conspiracy language"
        },
        {
            "text": "AMAZING breakthrough!!! You WONT BELIEVE what happens next!!!",
            "should_be": "Fake",
            "description": "Caps + exclamations + clickbait"
        },
        {
            "text": "The government announced new policy changes today.",
            "should_be": "Real",
            "description": "Simple factual statement"
        }
    ]
    
    fake_correct = 0
    real_correct = 0
    total_fake = sum(1 for t in tests if t["should_be"] == "Fake")
    total_real = sum(1 for t in tests if t["should_be"] == "Real")
    
    for i, test in enumerate(tests, 1):
        try:
            response = requests.post(api_url, json={"text": test["text"]}, timeout=5)
            if response.status_code == 200:
                result = response.json()
                prediction = result["prediction_label"]
                confidence = result["confidence"]
                
                correct = prediction == test["should_be"]
                if correct:
                    if test["should_be"] == "Fake":
                        fake_correct += 1
                    else:
                        real_correct += 1
                
                status = "✅ CORRECT" if correct else "❌ WRONG"
                
                print(f"{i}. {status} - Expected: {test['should_be']}, Got: {prediction} ({confidence*100:.1f}%)")
                print(f"   {test['description']}")
                print(f"   Text: {test['text'][:50]}{'...' if len(test['text']) > 50 else ''}")
                print()
            else:
                print(f"{i}. ❌ API Error: {response.status_code}")
        except Exception as e:
            print(f"{i}. ❌ Error: {e}")
    
    print("=" * 55)
    print("📊 RESULTS:")
    if total_fake > 0:
        fake_accuracy = (fake_correct / total_fake) * 100
        print(f"🔥 Fake Detection: {fake_correct}/{total_fake} ({fake_accuracy:.1f}%)")
    
    if total_real > 0:
        real_accuracy = (real_correct / total_real) * 100
        print(f"📰 Real Detection: {real_correct}/{total_real} ({real_accuracy:.1f}%)")
    
    overall = ((fake_correct + real_correct) / len(tests)) * 100
    print(f"🎯 Overall: {overall:.1f}%")
    
    if fake_correct == total_fake:
        print("🎉 PERFECT! All fake news detected!")
    elif fake_correct >= total_fake * 0.8:
        print("👍 EXCELLENT! Most fake news detected!")
    else:
        print("⚠️ Still missing some fake news...")

if __name__ == "__main__":
    try:
        ultra_test()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("💡 Make sure API is running at http://localhost:5000")