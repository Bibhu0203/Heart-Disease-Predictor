import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the saved model
model = pickle.load(open('models/heart_disease_model.pkl', 'rb'))

st.set_page_config(page_title="Heart Health AI", page_icon="❤️")

st.title("❤️ Heart Disease Predictor")
st.markdown("Use this AI tool to check for potential heart disease risk based on clinical parameters.")

# --- INPUT SECTION ---
with st.container():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", 1, 120, 50)
        sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x==1 else "Female")
        cp = st.selectbox("Chest Pain Type", options=[1, 2, 3, 4], help="1: Typical Angina, 2: Atypical, 3: Non-anginal, 4: Asymptomatic")
        trestbps = st.number_input("Resting BP (mmHg)", 50, 250, 120)

    with col2:
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120", options=[1, 0], format_func=lambda x: "True" if x==1 else "False")
        restecg = st.selectbox("Resting ECG", options=[0, 1, 2])
        thalach = st.number_input("Max Heart Rate", 50, 220, 150)

    with col3:
        exang = st.selectbox("Exercise Angina", options=[1, 0], format_func=lambda x: "Yes" if x==1 else "No")
        oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0)
        slope = st.selectbox("ST Slope", options=[1, 2, 3])
        ca = st.selectbox("Major Vessels (0-3)", options=[0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", options=[3, 6, 7], format_func=lambda x: {3:"Normal", 6:"Fixed", 7:"Reversible"}[x])

# --- PREDICTION ---
if st.button("Analyze Risk", use_container_width=True):
    # Data must be in same order as model was trained
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]

    st.divider()
    if prediction[0] == 1:
        st.error(f"⚠️ **High Risk Detected** (Confidence: {probability:.1%})")
        st.warning("Please consult a medical professional immediately.")
    else:
        st.success(f"✅ **Low Risk Detected** (Confidence: {1-probability:.1%})")
        st.info("Results are based on statistical patterns; maintain a healthy lifestyle!")