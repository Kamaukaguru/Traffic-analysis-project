from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import joblib

app = FastAPI()

# Mount the static directory
app.mount("/static", StaticFiles(directory="Images"), name="static")

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

@app.get("/", response_class=HTMLResponse)
async def get_home():
    return """
        <html>
        <head>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-image: url('/static/street.jpg'); /* Path to your local image */
                    background-size: cover;
                    background-position: center;
                    background-attachment: fixed;
                    margin: 0;
                    padding: 20px;
                }
                .container {
                    max-width: 600px;
                    margin: 0 auto;
                    background: rgba(255, 255, 255, 0.9); /* White background with some transparency */
                    padding: 20px;
                    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                    border-radius: 8px;
                }
                h2 {
                    color: #333333;
                    text-align: center;
                    margin-bottom: 20px;
                }
                label {
                    font-weight: bold;
                    display: block;
                    margin-bottom: 5px;
                    color: #555555;
                }
                input, select {
                    width: 100%;
                    padding: 10px;
                    margin-bottom: 15px;
                    border: 1px solid #cccccc;
                    border-radius: 5px;
                    box-sizing: border-box;
                    font-size: 14px;
                }
                input[type="submit"] {
                    background-color: #4CAF50;
                    color: white;
                    font-weight: bold;
                    cursor: pointer;
                    transition: background-color 0.3s ease;
                }
                input[type="submit"]:hover {
                    background-color: #45a049;
                }
                .result {
                    background-color: #e8f4ea;
                    padding: 10px;
                    border-radius: 5px;
                    margin-top: 20px;
                    font-size: 16px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Traffic Predictor</h2>
                <form method="post" action="/predict">
                    <label for="hour">Hour (0-23):</label>
                    <input type="number" id="hour" name="hour" min="0" max="23" required>

                    <label for="day_of_week">Day of Week (1-7):</label>
                    <input type="number" id="day_of_week" name="day_of_week" min="1" max="7" required>

                    <label for="temperature">Temperature (°C):</label>
                    <input type="number" step="0.1" id="temperature" name="temperature" required>

                    <label for="weather_condition">Weather Condition:</label>
                    <select id="weather_condition" name="weather_condition" required>
                        <option value="sunny">Sunny</option>
                        <option value="cloudy">Cloudy</option>
                        <option value="rainy">Rainy</option>
                        <option value="foggy">Foggy</option>
                    </select>

                    <input type="submit" value="Submit">
                </form>
            </div>
        </body>
        </html>
    """

@app.post("/predict", response_class=HTMLResponse)
async def predict_traffic(
    hour: int = Form(...),
    day_of_week: int = Form(...),
    temperature: float = Form(...),
    weather_condition: str = Form(...)
):
    # Input validation
    if not (0 <= hour <= 23):
        raise HTTPException(status_code=400, detail="Hour must be between 0 and 23.")
    if not (1 <= day_of_week <= 7):
        raise HTTPException(status_code=400, detail="Day of the week must be between 1 and 7.")

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

    # Map the congestion level to a descriptive label
    congestion_label = congestion_mapping.get(predicted_congestion_level, 'unknown')

    # Prepare input data for traffic counts prediction
    input_data_traffic = input_data_congestion.copy()
    input_data_traffic['congestion_level'] = predicted_congestion_level

    # Predict traffic counts and convert to integers
    predicted_vehicle_count = int(vehicle_model.predict(input_data_traffic)[0])
    predicted_pedestrian_count = int(pedestrian_model.predict(input_data_traffic)[0])

    # Return the predictions within the HTML form
    return f"""
        <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-image: url('/static/traffic.jpg'); /* Path to your local image */
                    background-size: cover;
                    background-position: center;
                    margin: 0;
                    padding: 20px;
                }}
                .container {{
                    max-width: 600px;
                    margin: 0 auto;
                    background: rgba(255, 255, 255, 0.9); /* White background with some transparency */
                    padding: 20px;
                    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                    border-radius: 8px;
                }}
                h2 {{
                    color: #333333;
                    text-align: center;
                    margin-bottom: 20px;
                }}
                label {{
                    font-weight: bold;
                    display: block;
                    margin-bottom: 5px;
                    color: #555555;
                }}
                input, select {{
                    width: 100%;
                    padding: 10px;
                    margin-bottom: 15px;
                    border: 1px solid #cccccc;
                    border-radius: 5px;
                    box-sizing: border-box;
                    font-size: 14px;
                }}
                input[type="submit"] {{
                    background-color: #4CAF50;
                    color: white;
                    font-weight: bold;
                    cursor: pointer;
                    transition: background-color 0.3s ease;
                }}
                input[type="submit"]:hover {{
                    background-color: #45a049;
                }}
                .result {{
                    background-color: #e8f4ea;
                    padding: 10px;
                    border-radius: 5px;
                    margin-top: 20px;
                    font-size: 16px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Traffic Predictor</h2>
                <form method="post" action="/predict">
                    <label for="hour">Hour (0-23):</label>
                    <input type="number" id="hour" name="hour" min="0" max="23" value="{hour}" required>

                    <label for="day_of_week">Day of Week (1-7):</label>
                    <input type="number" id="day_of_week" name="day_of_week" min="1" max="7" value="{day_of_week}" required>

                    <label for="temperature">Temperature (°C):</label>
                    <input type="number" step="0.1" id="temperature" name="temperature" value="{temperature}" required>

                    <label for="weather_condition">Weather Condition:</label>
                    <select id="weather_condition" name="weather_condition" required>
                        <option value="sunny" {"selected" if weather_condition == "sunny" else ""}>Sunny</option>
                        <option value="cloudy" {"selected" if weather_condition == "cloudy" else ""}>Cloudy</option>
                        <option value="rainy" {"selected" if weather_condition == "rainy" else ""}>Rainy</option>
                        <option value="foggy" {"selected" if weather_condition == "foggy" else ""}>Foggy</option>
                    </select>

                    <input type="submit" value="Submit">
                </form>
                <div class="result">
                    <p><strong>Predicted Congestion Level:</strong> {congestion_label}</p>
                    <p><strong>Predicted Vehicle Count:</strong> {predicted_vehicle_count}</p>
                    <p><strong>Predicted Pedestrian Count:</strong> {predicted_pedestrian_count}</p>
                </div>
            </div>
        </body>
        </html>
    """
