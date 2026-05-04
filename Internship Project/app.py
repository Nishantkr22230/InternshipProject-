import streamlit as st
import numpy as np
import joblib

# Load trained models
diabetes_model = joblib.load("diabetes_model.pkl")
diabetes_scaler = joblib.load("diabetes_scaler.pkl")
heart_model = joblib.load("heart_model.pkl")

# Title
st.title("🩺 Disease Prediction System")

# Sidebar menu
option = st.sidebar.selectbox(
    "Select Prediction",
    ["Diabetes Prediction", "Heart Disease Prediction"]
)

# ===============================
# DIABETES PREDICTION
# ===============================
if option == "Diabetes Prediction":

    st.header("Diabetes Prediction")

    pregnancies = st.number_input("Pregnancies", min_value=0)
    glucose = st.number_input("Glucose Level")
    bp = st.number_input("Blood Pressure")
    skin = st.number_input("Skin Thickness")
    insulin = st.number_input("Insulin")
    bmi = st.number_input("BMI")
    dpf = st.number_input("Diabetes Pedigree Function")
    age = st.number_input("Age")

    if st.button("Predict Diabetes"):

        input_data = np.array([[pregnancies, glucose, bp, skin,
                                insulin, bmi, dpf, age]])

        input_data = diabetes_scaler.transform(input_data)

        prediction = diabetes_model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ Person is Diabetic")
        else:
            st.success("✅ Person is NOT Diabetic")


# ===============================
# HEART DISEASE PREDICTION
# ===============================
elif option == "Heart Disease Prediction":

    st.header("Heart Disease Prediction")

    age = st.number_input("Age")
    sex = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])
    cp = st.number_input("Chest Pain Type (0-3)")
    trestbps = st.number_input("Resting Blood Pressure")
    chol = st.number_input("Cholesterol")
    fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
    restecg = st.number_input("Rest ECG (0-2)")
    thalach = st.number_input("Max Heart Rate")
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("Oldpeak")
    slope = st.number_input("Slope (0-2)")
    ca = st.number_input("CA (0-4)")
    thal = st.number_input("Thal (0-3)")

    if st.button("Predict Heart Disease"):

        input_data = np.array([[age, sex, cp, trestbps, chol,
                                fbs, restecg, thalach,
                                exang, oldpeak, slope, ca, thal]])

        prediction = heart_model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ Heart Disease Detected")
        else:
            st.success("✅ No Heart Disease")
