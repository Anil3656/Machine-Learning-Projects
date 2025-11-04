import streamlit as st
import joblib
import numpy as np

# ==============================
# Load the Trained Model & Vectorizer
# ==============================
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ==============================
# Page Configuration
# ==============================
st.set_page_config(
    page_title="📰 Fake News Detector",
    page_icon="🚫",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==============================
# Custom CSS Styling
# ==============================
st.markdown("""
<style>
    body {
        background-color: #f5f7fa;
        font-family: "Segoe UI", sans-serif;
    }
    .title {
        text-align: center;
        color: #2c3e50;
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.3em;
    }
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.1rem;
        margin-bottom: 2em;
    }
    .stTextArea textarea {
        border-radius: 12px;
        border: 1px solid #ccc;
        font-size: 1.05rem;
        padding: 12px;
    }
    .result {
        text-align: center;
        font-weight: 600;
        font-size: 1.4rem;
        padding: 1.2em;
        border-radius: 15px;
        margin-top: 1.5em;
    }
    .fake {
        background-color: #ffe5e5;
        color: #d63031;
        border: 2px solid #d63031;
    }
    .real {
        background-color: #e5ffe9;
        color: #27ae60;
        border: 2px solid #27ae60;
    }
    .footer {
        text-align: center;
        color: #95a5a6;
        margin-top: 3em;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# Header Section
# ==============================
st.markdown("<h1 class='title'>📰 Fake News Detection App</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Enter a news headline or article to check if it's <b>Fake</b> or <b>Real</b></p>", unsafe_allow_html=True)

# ==============================
# User Input
# ==============================
user_input = st.text_area("🧾 Enter the news content below:", height=200, placeholder="Type or paste a news headline/article here...")

# ==============================
# Prediction Logic
# ==============================
if st.button("🔍 Analyze News"):
    if user_input.strip():
        # Transform input and predict probabilities
        transformed_input = vectorizer.transform([user_input])
        
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(transformed_input)[0]
            prediction = np.argmax(proba)
            confidence = round(np.max(proba) * 100, 2)
        else:
            prediction = model.predict(transformed_input)[0]
            confidence = None

        # Display result with confidence
        if prediction == 1 or prediction == "FAKE":
            st.markdown("<div class='result fake'> This news is classified as <b>FAKE</b></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result real'> This news appears to be <b>REAL</b></div>", unsafe_allow_html=True)

        # Confidence score
        if confidence:
            st.progress(confidence / 100)
            st.markdown(f"**Confidence:** {confidence}%")
    else:
        st.warning("⚠️ Please enter a news text to analyze.")

# ==============================
# Footer
# ==============================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='footer'>Built with  using Streamlit & Scikit-learn</p>", unsafe_allow_html=True)
st.markdown("<p class='footer'>© 2024 Fake News Detector Developed by @AK</p>", unsafe_allow_html=True)
