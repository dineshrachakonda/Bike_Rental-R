import pandas as pd
from pathlib import Path


# Project folder
BASE_DIR = Path(__file__).resolve().parents[1]

RAW_FILE = BASE_DIR / "data" / "raw" / "day.csv"
PROCESSED_FILE = BASE_DIR / "data" / "processed" / "cleaned_data.csv"


def load_data():
    return pd.read_csv(RAW_FILE)


def clean_data(df):
    # Convert date column
    df["dteday"] = pd.to_datetime(df["dteday"])

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Handle missing values
    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:
        df[column] = df[column].fillna(
            df[column].median()
        )

    return df


def main():
    print("Loading data...")

    df = load_data()

    print("Original shape:", df.shape)

    df = clean_data(df)

    print("Cleaned shape:", df.shape)

    PROCESSED_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        PROCESSED_FILE,
        index=False
    )

    print("Processed data saved to:")
    print(PROCESSED_FILE)


if __name__ == "__main__":
    main()