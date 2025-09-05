import streamlit as st
import joblib
import whois
import datetime
import re
import scipy
import pandas as pd
from urlextract import URLExtract
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests

# ====== PAGE CONFIG ======
st.set_page_config(
    page_title="🛡️ Phishing Email Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====== LOAD LOTTIE ANIMATIONS ======
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

animation_safe = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_jbrw3hcz.json")
animation_alert = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_touohxv0.json")
animation_loading = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_usmfx6bp.json")

# ====== HEADER STYLE ======
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 10px;
    background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
    color: white;
    border-radius: 10px;
    font-size: 28px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🛡️ Enhanced Phishing Email Detector</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-powered email threat analysis with live risk scoring</p>", unsafe_allow_html=True)

# ====== LOAD MODEL ======
@st.cache_resource
def load_model():
    try:
        data = joblib.load("model.pkl")
        return data
    except FileNotFoundError:
        st.error("❌ Model file not found! Please train the model first.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

model_data = load_model()
vectorizer = model_data["vectorizer"]
model = model_data["model"]
suspicious_keywords = model_data["suspicious_keywords"]
extractor = URLExtract()

enhanced_keywords = suspicious_keywords + [
    'winner', 'congratulations', 'free', 'prize', 'claim',
    'security alert', 'unauthorized', 'blocked', 'frozen',
    'immediate action', 'final notice', 'last chance',
    'discount', 'offer', 'deal', 'promotion', 'bonus'
]

# ====== HELPER FUNCTIONS ======
def extract_urls_regex(email_text):
    return re.findall(r'(https?://[^\s]+)', email_text)

def extract_domain(url):
    try:
        if url.startswith('http'):
            domain = url.split("//")[-1].split("/")[0]
        else:
            domain = url.split("/")[0]
        return domain
    except:
        return None

def analyze_url_suspicion(urls):
    suspicious_patterns = [
        'bit.ly', 'tinyurl.com', 'shorturl.at', 't.co',
        'suspicious', 'secure', 'verify', 'login', 'account',
        'bank', 'paypal', 'amazon', 'microsoft', 'google'
    ]
    return sum(1 for url in urls for pattern in suspicious_patterns if pattern in url.lower())

def get_domain_age(domain):
    try:
        w = whois.whois(domain)
        creation_date = w.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        if creation_date:
            return (datetime.datetime.now() - creation_date).days
        return None
    except:
        return None

def predict_email_enhanced(email_text):
    try:
        urls = extractor.find_urls(str(email_text))
        urls_regex = extract_urls_regex(email_text)
        all_urls = list(set(urls + urls_regex))
        
        num_links = len(all_urls)
        suspicious_count = sum(k in email_text.lower() for k in suspicious_keywords)
        enhanced_suspicious_count = sum(k in email_text.lower() for k in enhanced_keywords)
        found_keywords = [k for k in enhanced_keywords if k in email_text.lower()]
        
        urgency_words = ['urgent', 'immediate', 'act now', 'expires', 'suspend', 'final notice']
        urgency_score = sum(1 for word in urgency_words if word in email_text.lower())
        
        url_suspicion = analyze_url_suspicion(all_urls)
        
        email_length = len(email_text)
        exclamation_count = email_text.count('!')
        uppercase_ratio = sum(1 for c in email_text if c.isupper()) / len(email_text) if email_text else 0
        money_words = ['money', 'cash', 'dollar', '$', 'payment', 'transfer', 'wire']
        money_mentions = sum(1 for word in money_words if word in email_text.lower())
        
        text_vec = vectorizer.transform([email_text])
        combined_features = scipy.sparse.hstack([text_vec, [[num_links, suspicious_count]]])
        
        ml_prediction = model.predict(combined_features)[0]
        probabilities = model.predict_proba(combined_features)[0]
        ml_confidence = max(probabilities) * 100
        
        risk_factors = {
            'has_external_links': num_links > 0,
            'multiple_suspicious_keywords': enhanced_suspicious_count > 2,
            'urgency_language': urgency_score > 0,
            'suspicious_urls': url_suspicion > 0,
            'excessive_exclamations': exclamation_count > 2,
            'high_uppercase_ratio': uppercase_ratio > 0.1,
            'money_mentions': money_mentions > 0,
            'very_short_email': email_length < 50,
            'ml_says_phishing': ml_prediction == "Phishing Email"
        }
        
        risk_score = sum(risk_factors.values())
        total_possible_risks = len(risk_factors)
        risk_percentage = (risk_score / total_possible_risks) * 100
        
        if risk_score >= 4:
            final_prediction = "Phishing Email"
            final_confidence = max(85, ml_confidence + 20)
            decision_reason = f"High risk indicators ({risk_score}/{total_possible_risks})"
        elif risk_score <= 2:
            final_prediction = "Safe Email"
            final_confidence = max(75, ml_confidence + 15)
            decision_reason = f"Low risk indicators ({risk_score}/{total_possible_risks})"
        else:
            final_prediction = ml_prediction
            if ml_prediction == "Phishing Email" and risk_score >= 3:
                final_confidence = min(ml_confidence + 25, 95)
                decision_reason = "ML prediction + supporting evidence"
            elif ml_prediction == "Safe Email" and risk_score <= 3:
                final_confidence = min(ml_confidence + 20, 90)
                decision_reason = "ML prediction + low risk indicators"
            else:
                final_confidence = ml_confidence
                decision_reason = "ML prediction (moderate confidence)"
        
        return {
            'prediction': final_prediction,
            'confidence': final_confidence,
            'decision_reason': decision_reason,
            'risk_factors': risk_factors,
            'risk_score': risk_score,
            'risk_percentage': risk_percentage,
            'features': {
                'urls': all_urls,
                'num_links': num_links,
                'suspicious_count': enhanced_suspicious_count,
                'found_keywords': found_keywords,
                'urgency_score': urgency_score,
                'url_suspicion': url_suspicion,
                'money_mentions': money_mentions,
                'exclamation_count': exclamation_count,
                'uppercase_ratio': uppercase_ratio,
                'email_length': email_length
            }
        }
    except Exception as e:
        return {'error': str(e)}

# ====== INPUT ======
st.markdown("### ✉️ Paste Email Content Below")
email_text = st.text_area("", height=200, placeholder="Example: Dear user, your account is suspended. Click here to verify...")
st.caption(f"Characters typed: {len(email_text)}")

# ====== ANALYZE ======
if st.button("🚀 Scan Email", use_container_width=True):
    if email_text.strip():
        with st.spinner("🔍 Scanning for threats..."):
            st_lottie(animation_loading, height=150, key="loading_anim")
            result = predict_email_enhanced(email_text)

        if 'error' not in result:
            if result['prediction'] == "Phishing Email":
                st_lottie(animation_alert, height=150)
                st.error(f"🚨 PHISHING DETECTED ({result['confidence']:.1f}% Confidence)")
            else:
                st_lottie(animation_safe, height=150)
                st.success(f"✅ EMAIL SAFE ({result['confidence']:.1f}% Confidence)")

            st.markdown(f"**Decision Logic:** {result['decision_reason']}")

            # Risk Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result['risk_percentage'],
                title={'text': "Risk Score"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "red" if result['risk_percentage'] > 50 else "green"}}
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Risk Factors Table
            st.subheader("🔍 Risk Factors")
            risk_df = pd.DataFrame([
                {"Risk Factor": factor.replace('_', ' ').title(), "Present": "✅ Yes" if present else "❌ No"}
                for factor, present in result['risk_factors'].items()
            ])
            st.dataframe(risk_df, use_container_width=True)

        else:
            st.error(f"Error: {result['error']}")
    else:
        st.warning("⚠️ Please enter email content before scanning.")
