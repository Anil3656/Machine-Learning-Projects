import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
st.title("Heart Disease Prediction App")
st.markdown("### Predict the likelihood of heart disease using multiple Machine Learning models.")
st.write("---")

# -----------------------------
# MODEL LOADING SECTION
# -----------------------------
st.sidebar.header("Model Selection")

# Helper function to load model safely
def safe_load_model(filename):
    if os.path.exists(filename):
        return joblib.load(filename)
    else:
        st.sidebar.warning(f"Missing file: {filename}")
        return None

models = {
    'Logistic Regression': safe_load_model('logistic_regression.pkl'),
    'Decision Tree': safe_load_model('decision_tree.pkl'),
    'Random Forest': safe_load_model('random_forest.pkl'),
    'KNN': safe_load_model('knn.pkl'),
    'SVM': safe_load_model('svm.pkl'),
    'XGBoost': safe_load_model('xgboost.pkl'),
    'Naive Bayes': safe_load_model('naive_bayes.pkl'),
    'Gradient Boosting': safe_load_model('gradient_boosting.pkl'),
    'AdaBoost': safe_load_model('adaboost.pkl'),
    'Extra Trees': safe_load_model('extra_trees.pkl')
}

# Check essential files
try:
    scaler = joblib.load('scaler.pkl')
    imputer = joblib.load('imputer.pkl')
except Exception as e:
    st.error(f"Error loading scaler or imputer: {e}")
    st.stop()

# -----------------------------
# SIDEBAR SELECTION
# -----------------------------
available_models = [name for name, mdl in models.items() if mdl is not None]
if not available_models:
    st.error("No models were loaded. Please ensure .pkl files are in the same directory as this app.")
    st.stop()

model_choice = st.sidebar.selectbox("Choose a Model", available_models)

# -----------------------------
# INPUT FORM
# -----------------------------
st.header("Enter Patient Details Below")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=20, max_value=100, value=50)
    sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
    cp = st.number_input("Chest Pain Type (0–3)", min_value=0, max_value=3, value=0)
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", value=120)

with col2:
    chol = st.number_input("Cholesterol (mg/dl)", value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[("True", 1), ("False", 0)], format_func=lambda x: x[0])[1]
    restecg = st.number_input("Resting ECG Results (0–2)", min_value=0, max_value=2, value=1)
    thalach = st.number_input("Max Heart Rate Achieved", value=150)

with col3:
    exang = st.selectbox("Exercise Induced Angina", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
    oldpeak = st.number_input("ST Depression Induced by Exercise", value=1.0)
    slope = st.number_input("Slope of Peak Exercise ST Segment (0–2)", min_value=0, max_value=2, value=1)
    ca = st.number_input("Number of Major Vessels (0–4)", min_value=0, max_value=4, value=0)

thal = st.selectbox("Thal (0=Normal, 1=Fixed Defect, 2=Reversible Defect)", [0, 1, 2])

# Prepare input dataframe
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal]
})

# -----------------------------
# PREDICTION SECTION
# -----------------------------
st.write("---")
if st.button("Predict"):
    try:
        X_input = imputer.transform(input_data)
        X_input_scaled = scaler.transform(X_input)
        model = models[model_choice]

        prediction = model.predict(X_input_scaled)
        prob = model.predict_proba(X_input_scaled)[0][1] if hasattr(model, "predict_proba") else None

        st.subheader("Prediction Result:")
        if prediction[0] == 1:
            st.error("The patient **is likely to have heart disease.**")
        else:
            st.success("The patient **is unlikely to have heart disease.**")

        if prob is not None:
            st.info(f"**Model Confidence:** {prob * 100:.2f}%")

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# -----------------------------
# FOOTER
# -----------------------------
st.write("---")
st.caption(" Developed with  using Streamlit & Scikit-learn By AK!")
