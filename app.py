import streamlit as st
import numpy as np
import joblib

# Load models
diabetes_model = joblib.load("model/diabetes_model.sav")
heart_model = joblib.load("model/heart_disease_model.sav")
parkinsons_model = joblib.load("model/parkinsons_model.sav")

# --- Title ---
st.markdown("<h1 style='color:red;'>Health Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='color:blue; font-style:italic;'>Using Machine Learning</h3>", unsafe_allow_html=True)

# Sidebar
model_choice = st.sidebar.selectbox(
    "Choose a Prediction Model",
    ("Diabetes", "Heart Disease", "Parkinsons")
)

# --- Functions ---
def predict_diabetes(data):
    arr = np.array(data).reshape(1, -1)
    pred = diabetes_model.predict(arr)
    return "Diabetic" if pred[0] == 1 else "Not Diabetic"

def predict_heart(data):
    arr = np.array(data).reshape(1, -1)
    pred = heart_model.predict(arr)
    return "Heart Disease" if pred[0] == 1 else "No Heart Disease"

def predict_parkinsons(data):
    arr = np.array(data).reshape(1, -1)
    pred = parkinsons_model.predict(arr)
    return "Parkinsons" if pred[0] == 1 else "No Parkinsons"

# --- App Interface ---
if model_choice == "Diabetes":
    st.header("Diabetes Prediction")
    pregnancies = st.number_input("Number of Pregnancies", 0, 20)
    glucose = st.number_input("Glucose Level", 0, 200)
    bp = st.number_input("Blood Pressure", 0, 140)
    skin = st.number_input("Skin Thickness", 0, 100)
    insulin = st.number_input("Insulin Level", 0, 900)
    bmi = st.number_input("BMI", 0.0, 70.0)
    pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 2.5)
    age = st.number_input("Age", 0, 120)

    if st.button("Predict"):
        result = predict_diabetes([pregnancies, glucose, bp, skin, insulin, bmi, pedigree, age])
        st.markdown(f"<h2 style='color:red;'>Prediction: {result}</h2>", unsafe_allow_html=True)

# --- Developer Credit ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:green; font-size:14px;'>Developed by: Sudeep</p>", unsafe_allow_html=True)
