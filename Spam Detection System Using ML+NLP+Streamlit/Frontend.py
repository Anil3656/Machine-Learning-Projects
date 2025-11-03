import streamlit as st
import joblib
import numpy as np

# === Load Model and Vectorizer ===
model = joblib.load(
    r"C:\Users\aneel.kumar\OneDrive - IMCS Group\Desktop\Aneel\Naresh_IT\Spam_Detection_system\spam_detector.pkl"
)
vectorizer = joblib.load(
    r"C:\Users\aneel.kumar\OneDrive - IMCS Group\Desktop\Aneel\Naresh_IT\Spam_Detection_system\vectorizer.pkl"
)

# === Page Configuration ===
st.set_page_config(
    page_title="📩 Spam Detector",
    page_icon="🚫",
    layout="centered",
)

# === Custom CSS ===
st.markdown(
    """
    <style>
        body {
            background-color: #f4f6f8;
        }
        .main-title {
            font-size: 2.4rem;
            color: #222;
            text-align: center;
            font-weight: 700;
            margin-bottom: 5px;
        }
        .subtitle {
            font-size: 1.1rem;
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }
        .stTextArea textarea {
            border-radius: 12px;
            border: 1px solid #ccc;
            font-size: 1.05rem;
            padding: 10px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }
        .result-box {
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            font-size: 1.3rem;
            margin-top: 25px;
            font-weight: bold;
            transition: all 0.3s ease-in-out;
        }
        .spam {
            background-color: #ffe5e5;
            color: #d62828;
            border: 2px solid #d62828;
        }
        .ham {
            background-color: #e5ffe5;
            color: #2b9348;
            border: 2px solid #2b9348;
        }
        .confidence {
            text-align: center;
            color: #555;
            font-size: 1rem;
            margin-top: 8px;
        }
        .stButton button {
            background: linear-gradient(90deg, #007bff, #00c3ff);
            color: white !important;
            font-weight: 600;
            border-radius: 8px;
            height: 3rem;
            width: 100%;
            border: none;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        .stButton button:hover {
            transform: scale(1.03);
            transition: all 0.2s ease-in-out;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# === Sidebar Section ===
with st.sidebar:
    st.title("ℹ️ About the App")
    st.markdown(
        """
        **📩 Spam Message Detector**
        
        This app uses **Natural Language Processing (NLP)** to classify SMS or Email text as **Spam** or **Not Spam**.
        
        ---
        ### 🧠 Model Info
        - Algorithm: **Multinomial Naive Bayes**
        - Vectorization: **TF-IDF Vectorizer**
        - Accuracy: ~98%
        
        ---
        ### 📊 Dataset
        - Source: UCI SMS Spam Collection
        - Samples: 5,574 messages
        
        ---
        ### 👨‍💻 Developer
        **Aneel Kumar Muppana**
        - 📧aneelmuppanakumar636@gmail.com
        - 🔗 [LinkedIn](https://www.linkedin.com/in/aneelkumar1927/)
        ---
        💡 *Try messages like:*  
        `"Congratulations! You won a lottery!"`  
        `"Can we meet tomorrow at 10 AM?"`
        """
    )

# === App Header ===
st.markdown("<h1 class='main-title'>📩 Spam Message Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Classify your SMS or Email as <b>Spam</b> or <b>Not Spam</b> instantly 🚀</p>", unsafe_allow_html=True)

# === Input Section ===
user_input = st.text_area("✉️ Enter your message below:", height=150, placeholder="Type your SMS or email message here...")

# === Predict Button ===
if st.button("🔍 Analyze Message"):
    if user_input.strip():
        # Transform input
        input_data = vectorizer.transform([user_input])
        prediction = model.predict(input_data)[0]

        # Get prediction probabilities (if available)
        try:
            proba = model.predict_proba(input_data)[0]
            confidence = np.max(proba) * 100
        except Exception:
            confidence = None

        # Handle both numeric and string labels
        label = str(prediction).lower().strip()

        if label in ["spam", "1"]:
            st.markdown("<div class='result-box spam'>This message is classified as <b>SPAM!</b></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-box ham'>This message is classified as <b>NOT SPAM</b></div>", unsafe_allow_html=True)

        # Display confidence score
        if confidence is not None:
            st.markdown(f"<p class='confidence'>Model Confidence: <b>{confidence:.2f}%</b></p>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a message to analyze.")

# === Footer ===
st.markdown("---")
st.markdown("<p style='text-align:center; color:#999;'>Built with using <b>Streamlit</b> & <b>Scikit-learn</b></p>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#999;'>Developed By  <b>Aneel Kumar M</b></p>", unsafe_allow_html=True)

