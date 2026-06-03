import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

st.title("🏠 House Price Predictor")

income = st.number_input("Median Income", value=5.0)
age = st.number_input("House Age", value=20.0)
rooms = st.number_input("Average Rooms", value=6.0)

if st.button("Predict Price"):

    features = np.array([[
        income,
        age,
        rooms,
        1.0,      # Avg Bedrooms
        1000.0,   # Population
        3.0,      # Avg Occupancy
        34.0,     # Latitude
        -118.0    # Longitude
    ]])

    prediction = model.predict(features)

    st.success(
        f"Predicted House Price: ${prediction[0]*100000:.2f}"
    )