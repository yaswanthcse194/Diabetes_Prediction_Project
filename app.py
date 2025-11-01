import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('diabetes_model.pkl')

# Streamlit app title
st.title("🩺 Diabetes Prediction App")
st.write("Enter your medical details below to predict the likelihood of having diabetes.")

# Input fields
pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose Level", min_value=0.0, max_value=300.0, value=100.0)
blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, max_value=200.0, value=70.0)
skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0)
insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0.0, max_value=1000.0, value=80.0)
bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input("Age (years)", min_value=0, max_value=120, value=30)

# Prediction button
if st.button("Predict"):
    # Prepare data for model
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    
    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Display result
    if prediction == 1:
        st.error(f"🚨 The person is **likely diabetic** (Probability: {probability:.2f})")
    else:
        st.success(f"✅ The person is **not diabetic** (Probability: {probability:.2f})")
