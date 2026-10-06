import joblib
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_FILE = BASE_DIR / "models" / "bike_rental_model.pkl"


def predict_rentals(data):

    model_data = joblib.load(MODEL_FILE)

    model = model_data["model"]
    features = model_data["features"]

    input_data = pd.DataFrame([data], columns=features)

    prediction = model.predict(input_data)

    return round(float(prediction[0]), 2)


if __name__ == "__main__":

    print("\nBike Rental Demand Prediction")
    print("------------------------------")

    season = int(input("Enter season (1-4): "))
    year = int(input("Enter year (0 or 1): "))
    month = int(input("Enter month (1-12): "))
    holiday = int(input("Holiday? (0/1): "))
    workingday = int(input("Working day? (0/1): "))
    weather = int(input("Weather situation (1-4): "))
    temperature = float(input("Temperature (0-1): "))
    feels_temperature = float(input("Feels temperature (0-1): "))
    humidity = float(input("Humidity (0-1): "))
    windspeed = float(input("Windspeed (0-1): "))
    weekday = int(input("Weekday (0-6): "))

    data = [
        season,
        year,
        month,
        holiday,
        workingday,
        weather,
        temperature,
        feels_temperature,
        humidity,
        windspeed,
        weekday
    ]

    result = predict_rentals(data)

    print("\nPredicted bike rentals:", result)