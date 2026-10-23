
from pathlib import Path
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "synthetic_healthcare_records_500.csv"
OUTPUT_DIR = PROJECT_DIR / "data"
OUTPUT_FILE = OUTPUT_DIR / "preprocessed_healthcare_records.csv"


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    return pd.read_csv(DATA_FILE)


def preprocess_data(df):
    df = df.copy()

    original_rows = len(df)

    # Remove completely duplicated rows
    df = df.drop_duplicates()

    # Standardize column names
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Clean text values without changing missing values
    for column in df.select_dtypes(include=["object"]).columns:
        df[column] = df[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    # Standardize common categorical columns if present
    for column in ["age_group", "category", "reported_symptoms"]:
        if column in df.columns:
            df[column] = df[column].map(
                lambda value: value.lower() if isinstance(value, str) else value
            )

    print("PREPROCESSING SUMMARY")
    print("-" * 35)
    print(f"Original rows: {original_rows}")
    print(f"Rows after removing duplicates: {len(df)}")
    print(f"Duplicate rows removed: {original_rows - len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Missing values: {int(df.isna().sum().sum())}")

    return df


def save_preprocessed_data(df):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nPreprocessed dataset saved to:\n{OUTPUT_FILE}")


def main():
    data = load_data()
    cleaned_data = preprocess_data(data)
    save_preprocessed_data(cleaned_data)


if __name__ == "__main__":
    main()
