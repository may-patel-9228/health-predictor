import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(page_title="Health Predictor", layout="wide")

st.title("🧠 Multi Disease Prediction System")

# Sidebar
page = st.sidebar.radio("Select Prediction", ["Heart Disease", "Diabetes"])

# ================= HEART =================
if page == "Heart Disease":
    st.header("❤️ Heart Disease Prediction")

    df = pd.read_csv("heart.csv")

    X = df.drop("target", axis=1)
    y = df["target"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = RandomForestClassifier()
    model.fit(X, y)

    age = st.number_input("Age", min_value=1, max_value=120)
    trestbps = st.number_input("Blood Pressure")
    chol = st.number_input("Cholesterol")
    thalach = st.number_input("Heart Rate")
    oldpeak = st.number_input("Oldpeak")
    ca = st.number_input("Vessels")

    if st.button("Predict Heart Disease"):
        data = np.array([[age, trestbps, chol, thalach, oldpeak, ca]])
        
        # Adjust to match dataset columns
        data = np.pad(data, ((0,0),(0, X.shape[1]-data.shape[1])), 'constant')
        
        data = scaler.transform(data)
        result = model.predict(data)

        if result[0] == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk")

# ================= DIABETES =================
elif page == "Diabetes":
    st.header("🩸 Diabetes Prediction")

    df = pd.read_csv("diabetes.csv")

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = RandomForestClassifier()
    model.fit(X, y)

    glucose = st.number_input("Glucose")
    bmi = st.number_input("BMI")
    age = st.number_input("Age")
    insulin = st.number_input("Insulin")

    if st.button("Predict Diabetes"):
        data = np.array([[glucose, bmi, age, insulin]])
        
        data = np.pad(data, ((0,0),(0, X.shape[1]-data.shape[1])), 'constant')
        
        data = scaler.transform(data)
        result = model.predict(data)

        if result[0] == 1:
            st.error("⚠️ Diabetes Detected")
        else:
            st.success("✅ No Diabetes")
