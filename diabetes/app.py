import streamlit as st
import joblib
import numpy as np


# Loading model and scaler
model = joblib.load("models/logistic_balanced_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.set_page_config(page_title="Diabetes Risk Prediction")

st.title("🩺 Diabetes Risk Prediction System")
st.write("Enter patient details below to assess diabetes risk.")


# Taking user input
preg = st.number_input("Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose Level (mg/dL)")
bp = st.number_input("Blood Pressure (mm Hg)")
skin = st.number_input("Skin Thickness (mm)")
insulin = st.number_input("Insulin Level (µU/mL)")
bmi = st.number_input("BMI")
age = st.number_input("Age", min_value=1, step=1)


# Checking family history for DPF
family_history = st.selectbox(
    "Family History of Diabetes",
    ["None", "One Parent", "Both Parents", "Siblings"]
)


# Mapping family history to approximate DPF value
if family_history == "None":
    dpf = 0.2
elif family_history == "One Parent":
    dpf = 0.8
elif family_history == "Both Parents":
    dpf = 1.5
else:
    dpf = 1.0


# Prediction Button
if st.button("Predict"):
    # log transform to insulin
    insulin = np.log1p(insulin)
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠ High Risk of Diabetes (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Low Risk of Diabetes (Probability: {probability:.2f})")
