from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


app = FastAPI(title="Customer Churn Prediction API")

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "model"
    / "churn_model.pkl"
)

pipeline = joblib.load(MODEL_PATH)

YesNo = Literal["Yes", "No"]
InternetServiceStatus = Literal["Yes", "No", "No internet service"]


class CustomerInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    customerID: str | None = None

    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0)

    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]

    OnlineSecurity: InternetServiceStatus
    OnlineBackup: InternetServiceStatus
    DeviceProtection: InternetServiceStatus
    TechSupport: InternetServiceStatus
    StreamingTV: InternetServiceStatus
    StreamingMovies: InternetServiceStatus

    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]

    MonthlyCharges: float = Field(ge=0, allow_inf_nan=False)
    TotalCharges: float | None = Field(
        ...,
        ge=0,
        allow_inf_nan=False
    )
def add_features(data):
    result = data.copy()

    service_columns = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    result["AdditionalServiceCount"] = (
        result[service_columns].eq("Yes").sum(axis=1)
    )

    automatic_methods = [
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]

    result["UsesAutomaticPayment"] = (
        result["PaymentMethod"].isin(automatic_methods).astype(int)
    )

    return result

@app.post("/predict")
def predict(customer: CustomerInput):
    data = customer.model_dump(exclude={"customerID"})

    no_phone = data["PhoneService"] == "No"
    no_phone_lines = data["MultipleLines"] == "No phone service"

    if no_phone != no_phone_lines:
        raise HTTPException(
            status_code=422,
            detail="MultipleLines must be consistent with PhoneService."
        )

    service_columns = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    no_internet = data["InternetService"] == "No"

    for column in service_columns:
        service_unavailable = data[column] == "No internet service"

        if no_internet != service_unavailable:
            raise HTTPException(
                status_code=422,
                detail=f"{column} must be consistent with InternetService."
            )

    customer_df = pd.DataFrame([data])

    customer_df["TotalCharges"] = pd.to_numeric(
        customer_df["TotalCharges"],
        errors="coerce"
    )

    prepared_customer = add_features(customer_df)

    prediction = int(pipeline.predict(prepared_customer)[0])

    classes = list(pipeline.named_steps["model"].classes_)
    churn_index = classes.index(1)

    churn_probability = float(
        pipeline.predict_proba(prepared_customer)[0, churn_index]
    )

    return {
        "prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": churn_probability
    }