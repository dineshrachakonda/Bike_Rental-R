from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import predict_rentals


app = FastAPI(
    title="Bike Rental Demand Prediction API",
    version="1.0"
)


class BikeRentalInput(BaseModel):

    season: int
    year: int
    month: int
    holiday: int
    workingday: int
    weather: int
    temperature: float
    feels_temperature: float
    humidity: float
    windspeed: float
    weekday: int


@app.get("/")
def home():

    return {
        "message": "Bike Rental Demand Prediction API"
    }


@app.post("/predict")
def predict(data: BikeRentalInput):

    input_data = [
        data.season,
        data.year,
        data.month,
        data.holiday,
        data.workingday,
        data.weather,
        data.temperature,
        data.feels_temperature,
        data.humidity,
        data.windspeed,
        data.weekday
    ]

    prediction = predict_rentals(
        input_data
    )

    return {
        "predicted_bike_rentals": prediction
    }