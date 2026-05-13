# Diabetes Risk Prediction System

A machine learning based web application for predicting diabetes risk using patient health data.

The project combines exploratory data analysis, feature engineering, model training, threshold optimization, explainable AI, and interactive deployment using Streamlit.

---

# Project Overview

Diabetes is one of the most common chronic diseases worldwide. Early prediction can help identify high-risk individuals and support preventive healthcare decisions.

This project builds a complete machine learning pipeline for diabetes risk prediction using structured patient health records. Multiple classification algorithms were trained and evaluated, with CatBoost selected as the final production model based on performance and stability.

The deployed application allows users to:

- Enter patient information manually
- Load random patient samples from the dataset
- Predict diabetes risk probability
- View threshold-based classification results
- Understand predictions using SHAP explainability

---

# Key Features

- End-to-end machine learning workflow
- Exploratory Data Analysis (EDA)
- Data preprocessing pipeline
- Multiple model comparison
- Cross-validation evaluation
- Threshold optimization
- SHAP explainability
- Streamlit web application
- Random patient testing mode
- Manual patient input mode
- Interactive prediction confidence display

---

# Machine Learning Workflow

## 1. Data Preprocessing

The preprocessing pipeline includes:

- Handling categorical features
- One-hot encoding
- Feature scaling
- Train-test split
- Stratified validation

---

## 2. Models Evaluated

The following machine learning models were trained and compared:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Extra Trees Classifier
- Voting Classifier
- Stacking Classifier
- CatBoost Classifier

---

## 3. Final Model Selection

CatBoost was selected as the final model because of:

- Strong recall performance
- Better handling of categorical features
- Stable cross-validation performance
- Better probability calibration
- Good explainability with SHAP

---

## 4. Threshold Optimization

Instead of using the default probability threshold of `0.50`, the final system uses an optimized threshold to improve classification quality and balance precision-recall tradeoffs.

---

## 5. Explainable AI with SHAP

SHAP (SHapley Additive exPlanations) is used to explain local predictions.

The application provides SHAP waterfall plots showing:

- Which features increased diabetes risk
- Which features reduced diabetes risk
- Magnitude of feature contributions

This improves transparency and interpretability of the model.

---

# Project Structure

```text
Diabetes_Disease_Prediction/
│
├── streamlit_app.py
├── requirements.txt
├── requirements-dev.txt
├── README.md
├── .gitignore
│
├── models/
│   ├── catboost_diabetes_model.pkl
│   └── threshold.pkl
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── test_data.csv
│
└── notebooks/
    ├── 01_diabetes_eda.ipynb
    ├── 02_diabetes_preprocessing.ipynb
    ├── 03_diabetes_modeling.ipynb
    ├── 04_dataset_comparison.ipynb
    ├── 05_diabetes_eda_2.0.ipynb
    ├── 06_diabetes_modeling_2.0.ipynb
    └── 07_pima_dataset_validation_2.0.ipynb
```

---

# Technologies Used

## Programming Language

- Python

## Data Analysis

- Pandas
- NumPy

## Visualization

- Matplotlib
- Seaborn
- Plotly

## Machine Learning

- Scikit-learn
- CatBoost

## Explainable AI

- SHAP

## Deployment

- Streamlit

---

# Installation

## 1. Clone Repository

```bash
git clone <your-repository-url>
cd Diabetes_Disease_Prediction
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Runtime Dependencies

The `requirements.txt` file contains the minimal dependencies required to run the Streamlit application.

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run streamlit_app.py
```

This setup is sufficient if you only want to:

- run the web application
- test predictions
- use the trained model
- explore the deployed interface

---

## 4. Install Development Dependencies

If you want to perform development-related tasks such as:

- model training
- exploratory data analysis
- notebook experimentation
- feature engineering
- visualization
- testing additional models

then install the additional development dependencies:

```bash
pip install -r requirements-dev.txt
```

The `requirements-dev.txt` file includes tools used during development, such as Jupyter Notebook and additional analysis libraries.

---

## 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

---

# Application Modes

## Random Patient Mode

Allows testing on random samples from the dataset.

Available options:

- Random Patient
- Non-Diabetic Patient
- Diabetic Patient

---

## Manual Input Mode

Users can manually provide:

- Gender
- Age
- Hypertension
- Heart Disease
- Smoking History
- BMI
- HbA1c Level
- Blood Glucose Level

The application then predicts diabetes risk probability and classification.

---

# Model Explainability

Each prediction includes SHAP-based local explanations.

The SHAP waterfall plot shows how individual features contributed to the final prediction.

Common influential features include:

- Blood glucose level
- HbA1c level
- BMI
- Age

---

# Important Notes

- This project is intended for educational and research purposes only.
- The application should not be used for real medical diagnosis.
- Clinical decisions should always involve qualified healthcare professionals.

---

# Future Improvements

Possible future enhancements:

- Integration with a cloud database for storing prediction history
- User authentication and patient profile management
- REST API deployment for external application integration
- Docker-based containerized deployment
- Mobile-friendly responsive interface
- Support for real-time clinical data integration
- Multi-disease risk prediction within a unified healthcare dashboard
- Advanced analytics and visualization dashboard for healthcare monitoring

---

# Author

**Rakib Islam**

B.Tech Computer Science & Engineering

---

# License

This project is open-source and available for educational purposes.
