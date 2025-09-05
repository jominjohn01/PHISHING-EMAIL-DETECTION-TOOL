# 🔧 Debug Script - Run this to diagnose prediction issues

import pandas as pd
import joblib
import scipy
from urlextract import URLExtract

# Initialize
extractor = URLExtract()

def debug_prediction_issues():
    """Comprehensive debugging for prediction problems"""
    
    print("🔍 PHISHING EMAIL DETECTION - DEBUGGING ANALYSIS")
    print("="*60)
    
    try:
        # Load model and check what it learned
        data = joblib.load("model.pkl")
        vectorizer = data["vectorizer"]
        model = data["model"]
        suspicious_keywords = data["suspicious_keywords"]
        
        print(f"✅ Model loaded successfully!")
        print(f"📊 Model classes: {model.classes_}")
        print(f"🔍 Suspicious keywords: {suspicious_keywords}")
        print()
        
        # Test with clear examples
        test_cases = [
            {
                "name": "🚨 OBVIOUS PHISHING EMAIL",
                "text": """
                URGENT! Your account will be suspended immediately!
                
                Click here to verify your password and login credentials:
                http://fake-bank-site.com/login
                
                Act now or lose access forever!
                """
            },
            {
                "name": "✅ OBVIOUS SAFE EMAIL", 
                "text": """
                Hi John,
                
                Thanks for attending today's meeting. The quarterly report 
                is attached for your review.
                
                Best regards,
                Sarah from Marketing
                """
            },
            {
                "name": "🟡 BORDERLINE EMAIL",
                "text": """
                Hello,
                
                Please update your account information at our website.
                We have made improvements to our security system.
                
                Thank you for your cooperation.
                """
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            print(f"\n{test_case['name']}")
            print("-" * 50)
            analyze_single_email(test_case['text'], vectorizer, model, suspicious_keywords)
    
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        print("💡 Make sure you ran 'python main.py' first!")

def analyze_single_email(email_text, vectorizer, model, suspicious_keywords):
    """Analyze a single email with detailed output"""
    
    # Extract features
    urls = extractor.find_urls(str(email_text))
    num_links = len(urls)
    suspicious_count = sum(k in str(email_text).lower() for k in suspicious_keywords)
    found_keywords = [k for k in suspicious_keywords if k in str(email_text).lower()]
    
    print(f"📧 Email preview: {email_text[:80].strip()}...")
    print(f"🔗 URLs found: {urls}")
    print(f"📊 Links count: {num_links}")
    print(f"⚠️  Suspicious keywords found: {found_keywords}")
    print(f"🔢 Suspicious count: {suspicious_count}")
    
    # Vectorize and predict
    text_vec = vectorizer.transform([email_text])
    combined = scipy.sparse.hstack([text_vec, [[num_links, suspicious_count]]])
    
    # Get prediction
    prediction = model.predict(combined)[0]
    probabilities = model.predict_proba(combined)[0]
    
    print(f"🤖 Model prediction: '{prediction}'")
    print(f"📈 Probabilities: {dict(zip(model.classes_, probabilities))}")
    print(f"🎯 Confidence: {max(probabilities)*100:.1f}%")
    
    # Check if prediction makes sense
    expected_phishing_indicators = num_links > 0 or suspicious_count > 2
    print(f"🧠 Expected phishing indicators: {expected_phishing_indicators}")
    
    if prediction == "Safe Email" and expected_phishing_indicators:
        print("⚠️  WARNING: This looks like it should be flagged as phishing!")
    elif prediction == "Phishing Email" and not expected_phishing_indicators:
        print("⚠️  WARNING: This might be a false positive!")

def check_label_distribution():
    """Check the original dataset labels"""
    try:
        df = pd.read_csv("Phishing_Email.csv")
        print(f"\n📊 ORIGINAL DATASET ANALYSIS:")
        print(f"Total emails: {len(df)}")
        print(f"Label distribution:")
        print(df['Email Type'].value_counts())
        print()
        
        # Show sample phishing emails
        print("📧 SAMPLE PHISHING EMAILS FROM DATASET:")
        phishing_samples = df[df['Email Type'] == 'Phishing Email']['Email Text'].head(3)
        for i, email in enumerate(phishing_samples):
            print(f"{i+1}. {str(email)[:100]}...")
        
        print("\n📧 SAMPLE SAFE EMAILS FROM DATASET:")
        safe_samples = df[df['Email Type'] == 'Safe Email']['Email Text'].head(3)
        for i, email in enumerate(safe_samples):
            print(f"{i+1}. {str(email)[:100]}...")
            
    except FileNotFoundError:
        print("❌ Could not find Phishing_Email.csv")

# Run the debugging
if __name__ == "__main__":
    debug_prediction_issues()
    check_label_distribution()
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("1. Check if the model classes match your expectations")
    print("2. Verify that suspicious keywords are appropriate")
    print("3. Test with more diverse email examples")
    print("4. Consider retraining with better balanced data")
    print("5. Add more sophisticated features (email headers, sender reputation)")
    
    print(f"\n🔧 QUICK FIXES TO TRY:")
    print("1. Increase suspicious keyword count threshold")
    print("2. Add more phishing-specific keywords")
    print("3. Adjust model confidence thresholds")
    print("4. Use ensemble methods for better accuracy")