
from pathlib import Path
import pandas as pd

# Find the folder containing this Python file
project_folder = Path(__file__).resolve().parent

# Build the correct CSV path
file_path = project_folder / "data" / "creditcard.csv"

print("CSV path:", file_path)
print("File exists:", file_path.exists())

if not file_path.exists():
    raise FileNotFoundError(
        f"CSV file not found at: {file_path}"
    )

df = pd.read_csv(file_path)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nTransaction counts:")
print(df["Class"].value_counts())
