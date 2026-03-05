import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'models', 'heart_disease_model.pkl')

st.set_page_config(page_title="Heart Health AI", page_icon="❤️")

# Load Model
@st.cache_resource # Keeps model in memory for speed
def load_model():
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

model = load_model()

if model is None:
    st.error("Model file not found. Please ensure 'models/heart_disease_model.pkl' exists.")
    st.stop()

st.title("❤️ Heart Disease Predictor")
st.write("Enter the clinical details below to predict the risk of heart disease.")

# --- UI LAYOUT ---
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 100, 50)
    sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x==1 else "Female")
    cp = st.selectbox("Chest Pain Type", options=[1, 2, 3, 4], help="1: Typical, 2: Atypical, 3: Non-anginal, 4: Asymptomatic")
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[1, 0], format_func=lambda x: "True" if x==1 else "False")

with col2:
    restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", options=[1, 0], format_func=lambda x: "Yes" if x==1 else "No")
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope of Peak Exercise ST", options=[1, 2, 3])
    ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", options=[3, 6, 7], format_func=lambda x: {3:"Normal", 6:"Fixed", 7:"Reversible"}[x])

# --- PREDICTION ---
if st.button("Analyze Risk", use_container_width=True):
    # Features must be in the exact order as the CSV columns (excluding 'num')
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]

    st.markdown("---")
    if prediction[0] == 1:
        st.error(f"### ⚠️ High Risk Detected")
        st.write(f"The model is **{probability:.1%}** confident that heart disease is present.")
    else:
        st.success(f"### ✅ Low Risk Detected")
        st.write(f"The model is **{1-probability:.1%}** confident that heart disease is NOT present.")
