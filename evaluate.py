import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
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


def evaluate_model():

    df = pd.read_csv(DATA_FILE)

    model_data = joblib.load(
        MODEL_FILE
    )

    model = model_data["model"]
    features = model_data["features"]

    X = df[features]
    y = df["cnt"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    predictions = model.predict(
        X_test
    )

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

    print("\nModel Evaluation")
    print("--------------------")
    print("MAE :", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2  :", round(r2, 2))


if __name__ == "__main__":
    evaluate_model()