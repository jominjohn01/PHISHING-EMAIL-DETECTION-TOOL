# 🔧 Quick Fix for Prediction Issues

import joblib
import scipy
from urlextract import URLExtract

def create_improved_prediction_function():
    """Create an improved prediction function with better logic"""
    
    improved_app_code = '''
# IMPROVED PREDICTION FUNCTION - Replace in your app.py

def predict_email_improved(email_text):
    """Enhanced prediction with better logic and debugging"""
    try:
        # Load model data
        data = joblib.load("model.pkl")
        vectorizer = data["vectorizer"]
        model = data["model"]
        suspicious_keywords = data["suspicious_keywords"]
        
        # Extract features
        urls = extractor.find_urls(str(email_text))
        num_links = len(urls)
        suspicious_count = sum(k in str(email_text).lower() for k in suspicious_keywords)
        found_keywords = [k for k in suspicious_keywords if k in str(email_text).lower()]
        
        # Additional heuristics
        urgency_words = ['urgent', 'immediate', 'act now', 'expires', 'suspend']
        urgency_score = sum(1 for word in urgency_words if word in email_text.lower())
        
        # Check for suspicious patterns
        has_links = num_links > 0
        has_suspicious_keywords = suspicious_count > 1
        has_urgency = urgency_score > 0
        
        # Vectorize text
        text_vec = vectorizer.transform([email_text])
        combined_features = scipy.sparse.hstack([text_vec, [[num_links, suspicious_count]]])
        
        # Make prediction
        prediction = model.predict(combined_features)[0]
        probabilities = model.predict_proba(combined_features)[0]
        confidence = max(probabilities) * 100
        
        # Enhanced decision logic
        risk_factors = {
            'external_links': has_links,
            'suspicious_keywords': has_suspicious_keywords,
            'urgency_language': has_urgency,
            'ml_prediction': prediction == "Phishing Email",
            'low_confidence': confidence < 60
        }
        
        risk_score = sum(risk_factors.values())
        
        # Override prediction if strong indicators
        if risk_score >= 3 and prediction == "Safe Email":
            final_prediction = "Phishing Email"
            override_reason = "Strong phishing indicators detected"
            confidence = min(confidence + 20, 95)  # Boost confidence
        elif risk_score <= 1 and prediction == "Phishing Email":
            final_prediction = "Safe Email" 
            override_reason = "Weak phishing indicators"
            confidence = min(confidence + 15, 90)
        else:
            final_prediction = prediction
            override_reason = None
        
        return {
            'prediction': final_prediction,
            'confidence': confidence,
            'original_prediction': prediction,
            'override_reason': override_reason,
            'risk_factors': risk_factors,
            'risk_score': risk_score,
            'features': {
                'urls': urls,
                'num_links': num_links,
                'suspicious_count': suspicious_count,
                'found_keywords': found_keywords,
                'urgency_score': urgency_score
            }
        }
        
    except Exception as e:
        return {'error': str(e)}

# USAGE IN STREAMLIT APP:
if st.button("🔍 Analyze Email (Improved)"):
    if email_text.strip():
        result = predict_email_improved(email_text)
        
        if 'error' not in result:
            prediction = result['prediction']
            confidence = result['confidence']
            
            if prediction == "Phishing Email":
                st.error(f"🚨 PHISHING EMAIL DETECTED - {confidence:.1f}% confidence")
                if result['override_reason']:
                    st.warning(f"📋 Override reason: {result['override_reason']}")
            else:
                st.success(f"✅ EMAIL APPEARS SAFE - {confidence:.1f}% confidence")
            
            # Show detailed analysis
            st.write("**Risk Analysis:**")
            for factor, present in result['risk_factors'].items():
                status = "✅" if present else "❌"
                st.write(f"{status} {factor.replace('_', ' ').title()}")
            
            st.write(f"**Risk Score:** {result['risk_score']}/5")
            
    '''
    
    return improved_app_code

# Show the improved code
print("🔧 IMPROVED PREDICTION FUNCTION:")
print("="*50)
improved_code = create_improved_prediction_function()
print(improved_code)