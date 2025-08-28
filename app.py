import streamlit as st
import numpy as np
import joblib
model = joblib.load("disease_model.pkl")
scaler = joblib.load("scaler.pkl")
st.set_page_config(page_title="Disease Prediction App")
st.title("Disease Prediction (Machine Learning)")
st.write("Enter patient details to predict the possibility of Diabetes.")
pregnancies = st.number_input("Number of Pregnancies", 0, 20, 0)
glucose = st.number_input("Glucose Level", 0, 300, 120)
blood_pressure = st.number_input("Blood Pressure", 0, 200, 70)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin Level", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 0, 120, 25)
features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                      insulin, bmi, dpf, age]], dtype=float)
features_scaled = scaler.transform(features)
if st.button("Predict"):
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    if prediction == 1:
        st.error(f"Result: DISEASE DETECTED — Confidence: {probability:.2f}")
    else:
        st.success(f"Result: NO DISEASE — Confidence: {probability:.2f}")
