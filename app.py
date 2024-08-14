import streamlit as st
import joblib
import pandas as pd

# Load the trained models
vehicle_model = joblib.load('vehicle_model.pkl')
pedestrian_model = joblib.load('pedestrian_model.pkl')

# Streamlit app
st.title('Traffic Predictor')

# Input fields
hour = st.number_input('Hour', min_value=0, max_value=23, value=12)
day_of_week = st.number_input('Day of Week', min_value=0, max_value=6, value=0)
temperature = st.number_input('Temperature', value=25.0)
congestion_level = st.number_input('Congestion Level', min_value=0, max_value=10, value=5)
weather_condition = st.selectbox('Weather Condition', ['Clear', 'Rain', 'Snow', 'Fog', 'Other'])

# Predict button
if st.button('Predict'):
    # Prepare input data
    input_data = pd.DataFrame({
        'hour': [hour],
        'day_of_week': [day_of_week],
        'temperature': [temperature],
        'congestion_level': [congestion_level],
        'weather_condition': [weather_condition]
    })

    # Predict traffic counts
    predicted_vehicle_count = vehicle_model.predict(input_data)[0]
    predicted_pedestrian_count = pedestrian_model.predict(input_data)[0]

    # Display results
    st.write(f'Vehicle Count: {int(predicted_vehicle_count)}')
    st.write(f'Pedestrian Count: {int(predicted_pedestrian_count)}')
