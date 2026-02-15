import joblib
import numpy as np

model = joblib.load("../models/logistic_balanced_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

def predict_diabetes(data):
    data = np.array(data).reshape(1, -1)
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    probability = model.predict_proba(data_scaled)[0][1]
    return prediction[0], probability