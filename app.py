
from pathlib import Path
import pandas as pd
import streamlit as st
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fraud_model.joblib"

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection")
st.write("Machine Learning Project using Random Forest")

st.info(
    "This is an educational demonstration. "
    "Predictions are not proof of fraud."
)

# Load trained model
if not MODEL_PATH.exists():
    st.error("Model not found. Run python3 train.py first.")
    st.stop()

saved = joblib.load(MODEL_PATH)
model = saved["model"]
features = saved["features"]

st.subheader("Upload Transaction CSV")

uploaded_file = st.file_uploader(
    "Select a CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        # Remove actual answer column if present
        if "Class" in data.columns:
            data = data.drop(columns=["Class"])

        missing = [col for col in features if col not in data.columns]

        if missing:
            st.error(
                "Missing columns: " + ", ".join(missing)
            )
            st.stop()

        # Use the same feature order as training
        X = data[features].apply(pd.to_numeric, errors="raise")

        if X.isna().any().any():
            st.error("CSV contains missing values.")
            st.stop()

        if not X.map(lambda value: pd.notna(value) and abs(value) != float("inf")).all().all():
            st.error("CSV contains invalid or infinite values.")
            st.stop()

        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

        results = data.copy()
        results["Predicted_Class"] = predictions
        results["Fraud_Probability_Percent"] = (
            probabilities * 100
        ).round(2)

        results["Prediction"] = results[
            "Predicted_Class"
        ].map({
            0: "Likely Legitimate",
            1: "Potential Fraud - Review"
        })

        # Dashboard metrics
        total = len(results)
        flagged = int((predictions == 1).sum())

        col1, col2, col3 = st.columns(3)
        col1.metric("Transactions", total)
        col2.metric("Potential Fraud Flags", flagged)
        col3.metric(
            "Flagged Percentage",
            f"{flagged / total * 100:.2f}%" if total else "0.00%"
        )

        st.subheader("Prediction Results")
        st.dataframe(results, use_container_width=True)

        csv_output = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Results CSV",
            data=csv_output,
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )

    except (ValueError, TypeError) as error:
        st.error(f"Could not process CSV: {error}")
