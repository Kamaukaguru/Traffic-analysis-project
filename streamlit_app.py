import streamlit as st
import pandas as pd
import joblib

# Load the trained LightGBM models
vehicle_model = joblib.load('vehicle_model.pkl')
pedestrian_model = joblib.load('pedestrian_model.pkl')
congestion_model = joblib.load('congestion_model.pkl')  # Model for predicting congestion level

# Congestion level mapping
congestion_mapping = {
    0: 'low',
    1: 'medium',
    2: 'high'
}

# Set up the background image (without requiring Pillow)
def set_bg_image(image_path):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{image_path}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Encode the image in base64 format
import base64
def get_image_as_base64(image_file):
    with open(image_file, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Background image path
bg_image = get_image_as_base64("Images/street.jpg")
set_bg_image(bg_image)

st.title("Traffic Predictor")

# Collect inputs from the user
hour = st.number_input("Hour (0-23):", min_value=0, max_value=23)
day_of_week = st.number_input("Day of Week (1-7):", min_value=1, max_value=7)
temperature = st.number_input("Temperature (°C):", format="%.1f")
weather_condition = st.selectbox("Weather Condition:", ["sunny", "cloudy", "rainy", "foggy"])

if st.button("Predict"):
    # Prepare input data for congestion level prediction
    input_data_congestion = pd.DataFrame({
        'hour': [hour],
        'day_of_week': [day_of_week],
        'temperature': [temperature],
        'weather_condition': [weather_condition]
    })

    # Handle categorical features
    categorical_columns = ['weather_condition']
    input_data_congestion[categorical_columns] = input_data_congestion[categorical_columns].astype('category')

    # Predict congestion level
    predicted_congestion_level = int(congestion_model.predict(input_data_congestion)[0])
    congestion_label = congestion_mapping.get(predicted_congestion_level, 'unknown')

    # Prepare input data for traffic counts prediction
    input_data_traffic = input_data_congestion.copy()
    input_data_traffic['congestion_level'] = predicted_congestion_level

    # Predict traffic counts
    predicted_vehicle_count = int(vehicle_model.predict(input_data_traffic)[0])
    predicted_pedestrian_count = int(pedestrian_model.predict(input_data_traffic)[0])

    # Display the predictions
    st.subheader("Predicted Traffic Details:")
    st.write(f"**Congestion Level:** {congestion_label}")
    st.write(f"**Vehicle Count:** {predicted_vehicle_count}")
    st.write(f"**Pedestrian Count:** {predicted_pedestrian_count}")
