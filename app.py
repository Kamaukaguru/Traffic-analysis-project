import streamlit as st
import joblib
import pandas as pd

# Load the trained models
vehicle_model = joblib.load('vehicle_model.pkl')
pedestrian_model = joblib.load('pedestrian_model.pkl')

st.title("Traffic Predictor")

# User input
hour = st.number_input("Hour", min_value=0, max_value=23, value=10)
day_of_week = st.number_input("Day of Week", min_value=0, max_value=6, value=2)
temperature = st.number_input("Temperature", value=22.0)
congestion_level = st.number_input("Congestion Level", min_value=0, value=3)
weather_condition = st.selectbox("Weather Condition", ["sunny", "cloudy", "rainy"])

# Prepare input data
input_data = pd.DataFrame({
    'hour': [hour],
    'day_of_week': [day_of_week],
    'temperature': [temperature],
    'congestion_level': [congestion_level],
    'weather_condition': [weather_condition]
})
input_data['weather_condition'] = input_data['weather_condition'].astype('category')

# Predict traffic counts
predicted_vehicle_count = vehicle_model.predict(input_data)[0]
predicted_pedestrian_count = pedestrian_model.predict(input_data)[0]

# Display the results
st.write(f"Vehicle Count: {predicted_vehicle_count}")
st.write(f"Pedestrian Count: {predicted_pedestrian_count}")
