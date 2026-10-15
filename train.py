
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    average_precision_score
)

# 1. Find project folder and dataset
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
MODEL_DIR = BASE_DIR / "models"

# 2. Load dataset
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

# 3. Check required target column
if "Class" not in df.columns:
    raise ValueError("Dataset must contain a Class column.")

if df["Class"].isna().any():
    raise ValueError("Class column contains missing values.")

if set(df["Class"].unique()) != {0, 1}:
    raise ValueError("Class must contain labels 0 and 1.")

# Separate input features and target
X = df.drop(columns=["Class"])
y = df["Class"].astype(int)

# Check numeric data and missing values
X = X.apply(pd.to_numeric, errors="raise")

if X.isna().any().any():
    raise ValueError("Input data contains missing values.")

if not all(pd.api.types.is_numeric_dtype(X[c]) for c in X):
    raise ValueError("All input features must be numeric.")

features = X.columns.tolist()

print("\nTransaction counts:")
print(y.value_counts().sort_index())

# 4. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# 5. Create the model
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

# 6. Train the model
print("\nModel training started...")
model.fit(X_train, y_train)
print("Model training completed!")

# 7. Predict the test data
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# 8. Evaluate the model
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Legitimate", "Fraud"],
        zero_division=0
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

print(
    "\nAverage Precision:",
    round(average_precision_score(y_test, probabilities), 4)
)

# 9. Save the trained model
MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {"model": model, "features": features},
    MODEL_DIR / "fraud_model.joblib"
)

print("\nModel saved successfully!")

# 10. Save confusion matrix graph
ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    labels=[0, 1],
    display_labels=["Legitimate", "Fraud"]
)

plt.title("Credit Card Fraud Detection")
plt.tight_layout()
plt.savefig(BASE_DIR / "confusion_matrix.png", dpi=150)
plt.close()

print("Confusion matrix graph saved!")
print("\nTraining process completed successfully.")
