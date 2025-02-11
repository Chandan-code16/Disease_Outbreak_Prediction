import streamlit as st
import numpy as np
import pickle
import os

# Set up Streamlit page configuration
st.set_page_config(page_title="Health Prediction App", page_icon="🏥", layout="wide")

# Load models
model_dir = "models"
diabetes_model = pickle.load(open(os.path.join(model_dir, "diabetes_model.sav"), "rb"))
heart_model = pickle.load(open(os.path.join(model_dir, "heart_model.sav"), "rb"))
parkinsons_model = pickle.load(open(os.path.join(model_dir, "parkinsons_model.sav"), "rb"))

# Sidebar menu
st.sidebar.title("Select Disease")
option = st.sidebar.radio("Choose an option:", ["Diabetes", "Heart Disease", "Parkinson's"])

# Diabetes Prediction
if option == "Diabetes":
    st.title("Diabetes Prediction")
    pregnancies = st.number_input("Pregnancies")
    glucose = st.number_input("Glucose")
    blood_pressure = st.number_input("Blood Pressure")
    skin_thickness = st.number_input("Skin Thickness")
    insulin = st.number_input("Insulin")
    bmi = st.number_input("BMI")
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function")
    age = st.number_input("Age")

    if st.button("Predict Diabetes"):
        user_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
        result = diabetes_model.predict(user_input)
        st.success("Diabetic" if result[0] == 1 else "Not Diabetic")

# Heart Disease Prediction
if option == "Heart Disease":
    st.title("Heart Disease Prediction")
    age = st.number_input("Age")
    sex = st.number_input("Sex (0 = Female, 1 = Male)")
    chest_pain = st.number_input("Chest Pain Type")
    resting_bp = st.number_input("Resting Blood Pressure")
    cholesterol = st.number_input("Cholesterol Level")
    fasting_blood_sugar = st.number_input("Fasting Blood Sugar")
    rest_ecg = st.number_input("Resting ECG Result")
    max_heart_rate = st.number_input("Max Heart Rate")

    if st.button("Predict Heart Disease"):
        user_input = np.array([[age, sex, chest_pain, resting_bp, cholesterol, fasting_blood_sugar, rest_ecg, max_heart_rate]])
        result = heart_model.predict(user_input)
        st.success("Heart Disease Detected" if result[0] == 1 else "No Heart Disease")

# Parkinson's Disease Prediction
if option == "Parkinson's":
    st.title("Parkinson's Disease Prediction")
    fo = st.number_input("MDVP:Fo(Hz)")
    fhi = st.number_input("MDVP:Fhi(Hz)")
    flo = st.number_input("MDVP:Flo(Hz)")
    jitter_percent = st.number_input("Jitter(%)")
    shimmer = st.number_input("Shimmer")
    hnr = st.number_input("HNR")
    rpde = st.number_input("RPDE")
    dfa = st.number_input("DFA")

    if st.button("Predict Parkinson's Disease"):
        user_input = np.array([[fo, fhi, flo, jitter_percent, shimmer, hnr, rpde, dfa]])
        result = parkinsons_model.predict(user_input)
        st.success("Parkinson's Detected" if result[0] == 1 else "No Parkinson's")