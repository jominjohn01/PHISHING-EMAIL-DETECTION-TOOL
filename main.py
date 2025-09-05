import pandas as pd
import whois
from urlextract import URLExtract
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import scipy
import joblib
from datetime import datetime
import csv
import os
import numpy as np

print("🚀 Starting Phishing Email Detection Model Training...")

# ====== STEP 1: Load Dataset ======
print("\n📊 Loading dataset...")
try:
    df = pd.read_csv("Phishing_Email.csv")  # Removed absolute path for portability
    print(f"✅ Dataset loaded successfully! Shape: {df.shape}")
    print(f"📈 Dataset preview:")
    print(df.head())
    print(f"\n📊 Class distribution:")
    print(df['Email Type'].value_counts())
except FileNotFoundError:
    print("❌ Error: Phishing_Email.csv not found!")
    print("Please ensure the dataset file is in the same directory as this script.")
    exit(1)

# ====== STEP 2: Data Preprocessing ======
print("\n🧹 Preprocessing data...")
# Handle missing values
df['Email Text'] = df['Email Text'].fillna('')
print(f"✅ Missing values handled")

# ====== STEP 3: Feature Extraction ======
print("\n🔍 Extracting features...")

# Initialize URL extractor
extractor = URLExtract()

# Extract URLs and count links
print("   - Extracting URLs...")
df['urls'] = df['Email Text'].apply(lambda x: extractor.find_urls(str(x)))
df['num_links'] = df['urls'].apply(len)

# Count suspicious keywords
print("   - Counting suspicious keywords...")
suspicious_keywords = ['urgent', 'verify', 'account', 'login', 'update', 'password', 
                      'click', 'confirm', 'suspend', 'limited', 'expire', 'act now']
df['suspicious_count'] = df['Email Text'].apply(
    lambda x: sum(k in str(x).lower() for k in suspicious_keywords)
)

# Text vectorization
print("   - Vectorizing text...")
vectorizer = CountVectorizer(max_features=2000, stop_words='english')
X_text = vectorizer.fit_transform(df['Email Text'].astype(str))

# Combine all features
print("   - Combining features...")
X = scipy.sparse.hstack([X_text, df[['num_links', 'suspicious_count']].values])
y = df['Email Type']  # "Phishing Email" / "Safe Email"

print(f"✅ Feature extraction complete!")
print(f"   - Total features: {X.shape[1]}")
print(f"   - Text features: {X_text.shape[1]}")
print(f"   - Additional features: 2 (num_links, suspicious_count)")

# ====== STEP 4: Train-Test Split ======
print("\n📊 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Data split complete!")
print(f"   - Training samples: {X_train.shape[0]}")
print(f"   - Testing samples: {X_test.shape[0]}")

# ====== STEP 5: Train Model ======
print("\n🤖 Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("✅ Model training complete!")

# ====== STEP 6: Model Evaluation ======
print("\n📈 Evaluating model performance...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"🎯 Model Accuracy: {accuracy:.4f}")
print(f"\n📊 Detailed Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print(f"\n🔍 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# ====== STEP 7: Save Model & Vectorizer ======
print("\n💾 Saving model and vectorizer...")
model_data = {
    "vectorizer": vectorizer,
    "model": model,
    "suspicious_keywords": suspicious_keywords,
    "feature_names": ["num_links", "suspicious_count"],
    "accuracy": accuracy
}
joblib.dump(model_data, "model.pkl")
print("✅ Model and vectorizer saved as model.pkl")

# ====== STEP 8: Feature Importance ======
print("\n🔍 Feature Importance Analysis...")
feature_names = vectorizer.get_feature_names_out().tolist() + ['num_links', 'suspicious_count']
importance_scores = model.feature_importances_
top_features = sorted(zip(feature_names, importance_scores), key=lambda x: x[1], reverse=True)

print("Top 10 Most Important Features:")
for i, (feature, importance) in enumerate(top_features[:10]):
    print(f"{i+1:2d}. {feature:20s}: {importance:.4f}")

# ====== STEP 9: WHOIS Domain Age Function ======
def check_domain_age(urls):
    """Check domain age for the first URL found"""
    try:
        if not urls:
            return 0
        domain = urls[0].split("/")[2] if len(urls[0].split("/")) > 2 else urls[0]
        domain_info = whois.whois(domain)
        creation_date = domain_info.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        if creation_date:
            age_days = (datetime.now() - creation_date).days
            return age_days
        else:
            return 0
    except Exception as e:
        print(f"⚠️ Error checking domain age: {str(e)}")
        return 0

# ====== STEP 10: Logging Function ======
def save_log(email_text, prediction, confidence, domain_age):
    """Save prediction logs to CSV file"""
    log_file = "scan_log.csv"
    file_exists = os.path.isfile(log_file)
    
    with open(log_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Timestamp", "Email Text (Preview)", "Prediction", "Confidence", "Domain Age (days)"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            email_text[:100] + "..." if len(email_text) > 100 else email_text,
            prediction,
            f"{confidence:.2f}%",
            domain_age
        ])

# ====== STEP 11: Single Email Prediction Function ======
def predict_email(text):
    """Predict single email - loads model from disk"""
    try:
        # Load model data
        data = joblib.load("model.pkl")
        vectorizer = data["vectorizer"]
        model = data["model"]
        suspicious_keywords = data["suspicious_keywords"]
        
        # Extract features
        urls = extractor.find_urls(text)
        num_links = len(urls)
        suspicious_count = sum(k in text.lower() for k in suspicious_keywords)
        
        # Vectorize text
        text_vec = vectorizer.transform([text])
        
        # Combine features
        combined = scipy.sparse.hstack([text_vec, [[num_links, suspicious_count]]])
        
        # Make prediction
        prediction = model.predict(combined)[0]
        probability = model.predict_proba(combined)[0]
        confidence = max(probability) * 100
        
        return prediction, confidence, urls
    except Exception as e:
        print(f"❌ Error in prediction: {str(e)}")
        return None, None, None

# ====== STEP 12: Batch Scan Function ======
def batch_scan(file_path):
    """Scan multiple emails from CSV file"""
    try:
        # Load model data
        data = joblib.load("model.pkl")
        vectorizer = data["vectorizer"]
        model = data["model"]
        suspicious_keywords = data["suspicious_keywords"]
        
        # Load batch data
        batch_df = pd.read_csv(file_path)
        results = []
        
        print(f"🔍 Scanning {len(batch_df)} emails...")
        
        for idx, email in enumerate(batch_df["Email Text"]):
            try:
                # Extract features
                urls = extractor.find_urls(str(email))
                domain_age = check_domain_age(urls)
                num_links = len(urls)
                suspicious_count = sum(k in str(email).lower() for k in suspicious_keywords)
                
                # Vectorize and predict
                text_vec = vectorizer.transform([str(email)])
                combined = scipy.sparse.hstack([text_vec, [[num_links, suspicious_count]]])
                
                prediction = model.predict(combined)[0]
                probability = model.predict_proba(combined)[0]
                confidence = max(probability) * 100
                
                # Save log
                save_log(str(email), prediction, confidence, domain_age)
                
                results.append({
                    'email_preview': str(email)[:50] + "...",
                    'prediction': prediction,
                    'confidence': f"{confidence:.2f}%",
                    'domain_age': domain_age
                })
                
                if (idx + 1) % 10 == 0:
                    print(f"   Processed {idx + 1} emails...")
                    
            except Exception as e:
                print(f"⚠️ Error processing email {idx + 1}: {str(e)}")
                continue
                
        return results
    except Exception as e:
        print(f"❌ Error in batch scan: {str(e)}")
        return []

# ====== STEP 13: Example Usage ======
if __name__ == "__main__":
    print("\n🧪 Testing single email prediction...")
    
    # Test email
    test_email = """
    URGENT: Your account will be suspended!
    
    Dear Customer,
    
    We have detected suspicious activity on your account. Please verify your login 
    credentials immediately by clicking the link below:
    
    http://suspicious-bank-site.com/verify-account
    
    Failure to act within 24 hours will result in account suspension.
    
    Thank you,
    Security Team
    """
    
    prediction, confidence, urls = predict_email(test_email)
    
    if prediction is not None:
        domain_age = check_domain_age(urls)
        save_log(test_email, prediction, confidence, domain_age)
        
        print(f"✅ Test Results:")
        print(f"   📧 Email Classification: {prediction}")
        print(f"   🎯 Confidence: {confidence:.2f}%")
        print(f"   🔗 URLs found: {len(urls)}")
        print(f"   📅 Domain age: {domain_age} days")
        print(f"   💾 Results saved to scan_log.csv")
    else:
        print("❌ Test failed!")

print(f"\n🎉 Training and setup complete!")
print(f"📁 Files created:")
print(f"   - model.pkl (trained model and vectorizer)")
print(f"   - scan_log.csv (prediction logs)")
print(f"\n🚀 You can now run the Streamlit app with: streamlit run app.py")