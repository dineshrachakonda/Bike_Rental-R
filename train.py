import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "features.csv"
)

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "bike_rental_model.pkl"
)


FEATURES = [
    "season",
    "yr",
    "mnth",
    "holiday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "weekday"
]

TARGET = "cnt"


def train_model():

    df = pd.read_csv(DATA_FILE)

    X = df[FEATURES]
    y = df[TARGET]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Create model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    # Train model
    model.fit(
        X_train,
        y_train
    )

    # Test model
    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    print("\nModel Results")
    print("--------------------")
    print("MAE :", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2  :", round(r2, 2))

    # Save model and feature names together
    model_data = {
        "model": model,
        "features": FEATURES
    }

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model_data,
        MODEL_FILE
    )

    print("\nModel saved to:")
    print(MODEL_FILE)


if __name__ == "__main__":
    train_model()