import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model and feature list
model = joblib.load("cibil_model.pkl")
model_features = joblib.load("model_features.pkl")

st.title("CIBIL Score Prediction App")

# Input form
with st.form("cibil_form"):
    AGE = st.slider("Age", 18, 100, 30)
    NETMONTHLYINCOME = st.number_input("Net Monthly Income", 0, 1000000, 30000)
    CC_utilization = st.slider("Credit Card Utilization", 0.0, 2.0, 0.5)
    PL_utilization = st.slider("Personal Loan Utilization", 0.0, 2.0, 0.5)
    CC_Flag = st.selectbox("Has Credit Card?", [0, 1])
    PL_Flag = st.selectbox("Has Personal Loan?", [0, 1])
    GENDER = st.selectbox("Gender", ['M', 'F'])
    EDUCATION = st.selectbox("Education", ['12TH', 'GRADUATE', 'POST-GRADUATE', 'SSC', 'UNDER GRADUATE'])
    MARITALSTATUS = st.selectbox("Marital Status", ['Single', 'Married'])
    submit = st.form_submit_button("Predict Credit Score")

# Mapping categorical inputs
gender_map = {'M': 1, 'F': 0}
edu_map = {
    '12TH': 0, 'SSC': 1, 'UNDER GRADUATE': 2,
    'GRADUATE': 3, 'POST-GRADUATE': 4
}
marital_map = {'Single': 0, 'Married': 1}

if submit:
    input_dict = {
        'AGE': AGE,
        'NETMONTHLYINCOME': NETMONTHLYINCOME,
        'CC_utilization': CC_utilization,
        'PL_utilization': PL_utilization,
        'CC_Flag': CC_Flag,
        'PL_Flag': PL_Flag,
        'GENDER': gender_map[GENDER],
        'EDUCATION': edu_map[EDUCATION],
        'MARITALSTATUS': marital_map[MARITALSTATUS],
    }

    # Fill missing features with -99999 or 0 as per your dataset
    full_input = {feat: input_dict.get(feat, -99999) for feat in model_features}
    X_input = pd.DataFrame([full_input])

    # Predict
    score = model.predict(X_input)[0]
    st.success(f"Predicted Credit Score: {round(score)}")
