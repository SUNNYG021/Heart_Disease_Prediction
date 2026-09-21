import pickle
import numpy as np
import streamlit as st

# Load the saved machine learning model
model = pickle.load(open('heart_model.sav', 'rb'))

st.title("❤️ Heart Disease Prediction App")

# Input fields with complete UX/UI tooltips
age = st.number_input("Age", min_value=1, max_value=120, value=30,
                      help="Patient's age in years.")

sex = st.selectbox("Sex", options=(0, 1), format_func=lambda x: "Female" if x == 0 else "Male") 

cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, value=0,
                     help="0 = Typical Angina (severe/classic), 1 = Atypical Angina, 2 = Non-anginal pain (not heart-related), 3 = Asymptomatic (no pain).")

trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120,
                           help="Resting blood pressure in mm Hg on admission to the hospital. Normal is around 120.")

chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200,
                       help="Serum cholesterol in mg/dl. Healthy levels are generally under 200.")

fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=(0, 1), format_func=lambda x: "Normal (<= 120)" if x == 0 else "High (> 120)",
                   help="A value greater than 120 mg/dl indicates potential diabetes, a major risk factor for heart disease.")

restecg = st.number_input("Resting ECG results (0-2)", min_value=0, max_value=2, value=1,
                          help="0 = Normal, 1 = ST-T wave abnormality, 2 = Probable or definite left ventricular hypertrophy (enlarged heart muscle).")

thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250, value=150,
                          help="The maximum heart rate the patient achieved during a stress test. Lower maximums can indicate weaker heart function.")

exang = st.selectbox("Exercise Induced Angina", options=(0, 1), format_func=lambda x: "No" if x == 0 else "Yes",
                     help="Does the patient experience chest pain directly triggered by physical exertion?")

oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0, 
                          help="ST depression induced by exercise relative to rest. A higher number indicates an abnormal heartbeat during exercise.")

slope = st.number_input("Slope (0-2)", min_value=0, max_value=2, value=1, 
                        help="The slope of the peak exercise ST segment. 0 = Upsloping (better heart rate), 1 = Flat (typical healthy), 2 = Downsloping (signs of an unhealthy heart).")

ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0, 
                     help="Number of major blood vessels colored by fluoroscopy. A higher number means more blood vessels are blocked.")

thal = st.number_input("Thal (0-3)", min_value=0, max_value=3, value=1, 
                       help="Thalassemia blood disorder. 1 = Normal, 2 = Fixed Defect (no blood flow in some parts), 3 = Reversible Defect (blood flow is observed but not normal).")

# Prediction Logic
if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                             exang, oldpeak, slope, ca, thal]])
    
    # Target 0 is the actual diseased class in this Kaggle dataset
    probabilities = model.predict_proba(input_data)
    risk_score = probabilities[0][0] * 100  
    
    st.subheader(f"Calculated Risk Score: {risk_score:.2f}%")
    
    if risk_score <= 30:
        st.success("🟢 Low Risk: You do not need to visit a doctor immediately. Maintain a balanced diet and regular physical exercise.")
    elif risk_score <= 70:
        st.warning("🟡 Medium Risk: You should schedule a routine consultation with a doctor and actively limit your daily salt intake.")
    else:
        st.error("🔴 High Risk: Immediate attention recommended! You should see a cardiologist immediately and strictly limit your salt and cholesterol intake.")