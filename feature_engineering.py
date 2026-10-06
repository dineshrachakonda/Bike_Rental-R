import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "cleaned_data.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "features.csv"
)


# Features used by the model
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


def create_features(df):
    # Create useful date features
    df["year"] = df["dteday"].dt.year
    df["month"] = df["dteday"].dt.month
    df["day"] = df["dteday"].dt.day
    df["weekday"] = df["dteday"].dt.weekday

    return df


def main():
    print("Loading cleaned data...")

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["dteday"]
    )

    df = create_features(df)

    # Keep only model features and target
    final_columns = FEATURES + [TARGET]

    df = df[final_columns]

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("Feature engineering completed.")
    print("Features:", FEATURES)
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()