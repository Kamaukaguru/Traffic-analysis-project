from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.staticfiles import StaticFiles

# Load the trained models
vehicle_model = joblib.load('vehicle_model.pkl')
pedestrian_model = joblib.load('pedestrian_model.pkl')

app = FastAPI()

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_home():
    with open('static/index.html', 'r') as file:
        return HTMLResponse(file.read())

@app.post("/predict", response_class=HTMLResponse)
async def predict_traffic(
    hour: int = Form(...),
    day_of_week: int = Form(...),
    temperature: float = Form(...),
    congestion_level: int = Form(...),
    weather_condition: str = Form(...)
):
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

    # Return the predictions within the HTML form
    return f"""
        <h2>Traffic Predictor</h2>
        <form method="post" action="/predict">
            <label for="hour">Hour:</label>
            <input type="text" id="hour" name="hour" value="{hour}">
            <br><br>

            <label for="day_of_week">Day of Week:</label>
            <input type="text" id="day_of_week" name="day_of_week" value="{day_of_week}">
            <br><br>

            <label for="temperature">Temperature:</label>
            <input type="text" id="temperature" name="temperature" value="{temperature}">
            <br><br>

            <label for="congestion_level">Congestion Level:</label>
            <input type="text" id="congestion_level" name="congestion_level" value="{congestion_level}">
            <br><br>

            <label for="weather_condition">Weather Condition:</label>
            <input type="text" id="weather_condition" name="weather_condition" value="{weather_condition}">
            <br><br>

            <input type="submit" value="Submit">
        </form>
        <br>
        <p>Vehicle Count: {predicted_vehicle_count}</p>
        <p>Pedestrian Count: {predicted_pedestrian_count}</p>
    """
