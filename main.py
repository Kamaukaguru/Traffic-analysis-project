from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
import pandas as pd
import joblib

app = FastAPI()

# Load the trained LightGBM models
vehicle_model = joblib.load('vehicle_model.pkl')
pedestrian_model = joblib.load('pedestrian_model.pkl')

@app.get("/", response_class=HTMLResponse)
async def get_home():
    return """
        <h2>Traffic Predictor</h2>
        <form method="post" action="/predict">
            <label for="hour">Hour (0-23):</label>
            <input type="number" id="hour" name="hour" min="0" max="23" required>
            <br><br>

            <label for="day_of_week">Day of Week (1-7):</label>
            <input type="number" id="day_of_week" name="day_of_week" min="1" max="7" required>
            <br><br>

            <label for="temperature">Temperature (°C):</label>
            <input type="number" step="0.1" id="temperature" name="temperature" required>
            <br><br>

            <label for="congestion_level">Congestion Level:</label>
            <select id="congestion_level" name="congestion_level" required>
                <option value="1">Low</option>
                <option value="2">Medium</option>
                <option value="3">High</option>
            </select>
            <br><br>

            <label for="weather_condition">Weather Condition:</label>
            <input type="text" id="weather_condition" name="weather_condition" required>
            <br><br>

            <input type="submit" value="Submit">
        </form>
    """

@app.post("/predict", response_class=HTMLResponse)
async def predict_traffic(
    hour: int = Form(...),
    day_of_week: int = Form(...),
    temperature: float = Form(...),
    congestion_level: int = Form(...),
    weather_condition: str = Form(...)
):
    # Input validation
    if not (0 <= hour <= 23):
        raise HTTPException(status_code=400, detail="Hour must be between 0 and 23.")
    if not (1 <= day_of_week <= 7):
        raise HTTPException(status_code=400, detail="Day of the week must be between 1 and 7.")
    if not (1 <= congestion_level <= 3):
        raise HTTPException(status_code=400, detail="Congestion level must be 1 (Low), 2 (Medium), or 3 (High).")

    # Prepare input data
    input_data = pd.DataFrame({
        'hour': [hour],
        'day_of_week': [day_of_week],
        'temperature': [temperature],
        'congestion_level': [congestion_level],
        'weather_condition': [weather_condition]
    })
    
    # Handle categorical features
    categorical_columns = ['weather_condition']
    input_data[categorical_columns] = input_data[categorical_columns].astype('category')

    # Predict traffic counts
    predicted_vehicle_count = vehicle_model.predict(input_data)[0]
    predicted_pedestrian_count = pedestrian_model.predict(input_data)[0]

    # Return the predictions within the HTML form
    return f"""
        <h2>Traffic Predictor</h2>
        <form method="post" action="/predict">
            <label for="hour">Hour (0-23):</label>
            <input type="number" id="hour" name="hour" min="0" max="23" value="{hour}" required>
            <br><br>

            <label for="day_of_week">Day of Week (1-7):</label>
            <input type="number" id="day_of_week" name="day_of_week" min="1" max="7" value="{day_of_week}" required>
            <br><br>

            <label for="temperature">Temperature (°C):</label>
            <input type="number" step="0.1" id="temperature" name="temperature" value="{temperature}" required>
            <br><br>

            <label for="congestion_level">Congestion Level:</label>
            <select id="congestion_level" name="congestion_level" required>
                <option value="1" {"selected" if congestion_level == 1 else ""}>Low</option>
                <option value="2" {"selected" if congestion_level == 2 else ""}>Medium</option>
                <option value="3" {"selected" if congestion_level == 3 else ""}>High</option>
            </select>
            <br><br>

            <label for="weather_condition">Weather Condition:</label>
            <input type="text" id="weather_condition" name="weather_condition" value="{weather_condition}" required>
            <br><br>

            <input type="submit" value="Submit">
        </form>
        <br>
        <p>Vehicle Count: {predicted_vehicle_count}</p>
        <p>Pedestrian Count: {predicted_pedestrian_count}</p>
    """

