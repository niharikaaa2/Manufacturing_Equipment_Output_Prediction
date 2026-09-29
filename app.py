
from fastapi import FastAPI
import pandas as pd
import joblib

# Load Model
model = joblib.load(
    "manufacturing_linear_regression.joblib"
)

# Create FastAPI App
app = FastAPI(
    title="Manufacturing Equipment Output Prediction",
    description="Linear Regression API for predicting Parts Per Hour",
    version="1.0"
)

# Home
@app.get("/")
def home():
    return {
        "message": "Manufacturing Output Prediction API",
        "status": "running"
    }

# Prediction Endpoint
@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    # Process timestamp
    if "Timestamp" in df.columns:

        df["Timestamp"] = pd.to_datetime(
            df["Timestamp"],
            errors="coerce"
        )

        df["Hour"] = df["Timestamp"].dt.hour
        df["Day"] = df["Timestamp"].dt.day
        df["Month"] = df["Timestamp"].dt.month

        df = df.drop(
            "Timestamp",
            axis=1
        )

    prediction = model.predict(df)

    return {
        "Predicted_Parts_Per_Hour":
        round(float(prediction[0]), 2)
    }
