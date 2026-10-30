import streamlit as st
import joblib
import numpy as np

st.title("🚕 Urban Taxi Fare Prediction")

# Load model safely
try:
    model = joblib.load("best_fare_model.pkl")
    model_loaded = True
except:
    st.error("❌ Model file is empty or corrupted. Please save the trained model again.")
    model_loaded = False

# Inputs
distance = st.number_input("Trip Distance (km)", min_value=0.1, step=0.1)
passenger_count = st.number_input("Passenger Count", min_value=1, max_value=6, step=1)
pickup_hour = st.number_input("Pickup Hour (0–23)", min_value=0, max_value=23, step=1)
pickup_day = st.selectbox("Pickup Day", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])

# Day encoding
day_mapping = { 
    "Monday":0, "Tuesday":1, "Wednesday":2,
    "Thursday":3, "Friday":4, "Saturday":5, "Sunday":6
}

pickup_day_encoded = day_mapping[pickup_day]

# Prediction
if st.button("Predict Fare"):
    if not model_loaded:
        st.error("❌ Cannot predict because model is missing or empty.")
    else:
        data = np.array([[distance, passenger_count, pickup_hour, pickup_day_encoded]])
        fare = model.predict(data)[0]
        st.success(f"Estimated Taxi Fare: ${fare:.2f}")
