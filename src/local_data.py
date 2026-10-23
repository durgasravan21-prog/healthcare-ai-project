from pathlib import Path
import pandas as pd

# Find the project folder
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Locate the dataset
DATA_FILE = PROJECT_DIR / "data" / "synthetic_healthcare_records_500.csv"

# Load the CSV file
df = pd.read_csv(DATA_FILE)

# Display basic information
print("Dataset loaded successfully!")
print("Number of records:", len(df))
print("Number of columns:", len(df.columns))

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five records:")
print(df.head())
