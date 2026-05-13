import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    layout="wide"
)

# ==========================================
# TITLE
# ==========================================

st.title("Diabetes Risk Prediction System")

st.write(
    "Machine Learning based diabetes risk prediction using "
    "CatBoost with SHAP explainability."
)

# ==========================================
# LOAD MODEL + DATA
# ==========================================
BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "models" / "catboost_diabetes_model.pkl")

threshold = joblib.load(BASE_DIR / "models" / "threshold.pkl")

test_data = pd.read_csv(BASE_DIR / "data" / "test_data.csv")

# SHAP Explainer
explainer = shap.TreeExplainer(model)

# ==========================================
# SIDEBAR INPUTS
# ==========================================

st.sidebar.header("Patient Input")

input_mode = st.sidebar.radio(
    "Choose Input Method",
    ["Random Patient", "Manual Input"]
)

# ==========================================
# RANDOM PATIENT MODE
# ==========================================

if input_mode == "Random Patient":

    patient_type = st.sidebar.radio(
    "Select Patient Type",
    ["Random", "Non-Diabetic", "Diabetic"]
    )

    if st.sidebar.button("Load Patient"):

        if patient_type == "Random":
            patient = test_data.sample(1)
        elif patient_type == "Non-Diabetic":
            patient = test_data[test_data["diabetes"] == 0].sample(1)
        else:
            patient = test_data[test_data["diabetes"] == 1].sample(1)

        patient = patient.reset_index(drop=True)

        input_df = patient.drop("diabetes", axis=1)

        actual_label = patient["diabetes"].values[0]

        # ==========================================
        # DISPLAY PATIENT DATA
        # ==========================================

        st.subheader("Patient Data")

        display_df = input_df.T.astype(str)

        st.dataframe(
            display_df,
            width="stretch"
        )

        # ==========================================
        # PREDICTION
        # ==========================================

        probability = model.predict_proba(input_df)[0][1]

        prediction = 1 if probability >= threshold else 0

        # ==========================================
        # RESULTS
        # ==========================================

        st.subheader("Prediction Result")

        st.subheader("Prediction Confidence")

        st.progress(float(probability))

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Raw Probability",
                f"{probability:.6f}"
            )

        with col2:
            st.metric(
                "Probability %",
                f"{probability * 100:.2f}%"
            )

        with col3:
            st.metric(
                "Optimized Threshold",
                f"{threshold:.2f}"
            )

        st.write(f"Actual Label: {actual_label}")

        # ==========================================
        # THRESHOLD-ALIGNED FINAL PREDICTION MESSAGE
        # ==========================================

        if prediction == 1:
            st.error("Model Prediction: Diabetic")
        else:
            st.success("Model Prediction: Non-Diabetic")

        # ==========================================
        # PREDICTION CORRECTNESS
        # ==========================================

        if prediction == actual_label:
            st.info("Prediction matched actual label.")
        else:
            st.warning("Prediction did not match actual label.")

        # ==========================================
        # SHAP LOCAL EXPLANATION
        # ==========================================

        st.subheader("SHAP Local Explanation")

        shap_values = explainer.shap_values(input_df)

        fig, ax = plt.subplots(figsize=(10, 5))

        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=input_df.iloc[0],
                feature_names=input_df.columns
            ),
            show=False
        )

        st.pyplot(fig)

# ==========================================
# MANUAL INPUT MODE
# ==========================================

else:

    gender = st.sidebar.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    age = st.sidebar.slider(
        "Age",
        1,
        100,
        30
    )

    hypertension = st.sidebar.selectbox(
        "Hypertension",
        [0, 1]
    )

    heart_disease = st.sidebar.selectbox(
        "Heart Disease",
        [0, 1]
    )

    smoking_history = st.sidebar.selectbox(
        "Smoking History",
        [
            "never",
            "former",
            "current",
            "not current",
            "No Info",
            "ever"
        ]
    )

    bmi = st.sidebar.slider(
        "BMI",
        10.0,
        60.0,
        25.0
    )

    hba1c = st.sidebar.slider(
        "HbA1c Level",
        3.0,
        15.0,
        5.0
    )

    glucose = st.sidebar.slider(
        "Blood Glucose Level",
        50,
        300,
        100
    )

    input_df = pd.DataFrame({
        "gender": [gender],
        "age": [age],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],
        "smoking_history": [smoking_history],
        "bmi": [bmi],
        "HbA1c_level": [hba1c],
        "blood_glucose_level": [glucose]
    })

    # ==========================================
    # DISPLAY PATIENT DATA
    # ==========================================

    st.subheader("Patient Data")

    display_df = input_df.T.astype(str)

    st.dataframe(
        display_df,
        width="stretch"
    )

    # ==========================================
    # PREDICT BUTTON
    # ==========================================

    if st.button("Predict Diabetes Risk"):

        # ==========================================
        # PREDICTION
        # ==========================================

        probability = model.predict_proba(input_df)[0][1]

        prediction = 1 if probability >= threshold else 0

        # ==========================================
        # RESULTS
        # ==========================================

        st.subheader("Prediction Result")

        st.subheader("Prediction Confidence")

        st.progress(float(probability))

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Raw Probability",
                f"{probability:.6f}"
            )

        with col2:
            st.metric(
                "Probability %",
                f"{probability * 100:.2f}%"
            )

        with col3:
            st.metric(
                "Optimized Threshold",
                f"{threshold:.2f}"
            )

        # ==========================================
        # THRESHOLD-ALIGNED FINAL PREDICTION MESSAGE
        # ==========================================

        if prediction == 1:
            st.error("Model Prediction: Diabetic")
        else:
            st.success("Model Prediction: Non-Diabetic")
        # ==========================================
        # SHAP LOCAL EXPLANATION
        # ==========================================

        st.subheader("SHAP Local Explanation")

        shap_values = explainer.shap_values(input_df)

        fig, ax = plt.subplots(figsize=(10, 5))

        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=input_df.iloc[0],
                feature_names=input_df.columns
            ),
            show=False
        )

        st.pyplot(fig)

# ==========================================
# DISCLAIMER
# ==========================================

st.warning(
    "This application is intended for educational and research purposes "
    "only. It should not be used for medical diagnosis."
)