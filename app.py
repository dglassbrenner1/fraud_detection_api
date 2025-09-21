from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
from typing import List

import mlflow

# Set the MLflow tracking URI to your Databricks tracking server
mlflow.set_tracking_uri("databricks")  # or the full HTTP URI of your tracking server


# Define expected input schema (adjust fields as per your model features)
class TransactionFeatures(BaseModel):
    TX_AMOUNT: float
    TX_DURING_WEEKEND: int
    TX_DURING_NIGHT: int
    Cust_Nb_Tx_1Day: int
    Cust_Avg_Amt_1Day: float
    Cust_Nb_Tx_7Day: int
    Cust_Avg_Amt_7Day: float
    Cust_Nb_Tx_30Day: int
    Cust_Avg_Amt_30Day: float
    Term_Nb_Tx_1Day: int
    Term_Risk_1Day: int
    Term_Nb_Tx_7Day: int
    Term_Risk_7Day: int
    Term_Nb_Tx_30Day: int
    Term_Risk_30Day: int

app = FastAPI()

# Load the model at startup (adjust model URI/version accordingly)
model_uri = "models:/fraud_detection_pipeline_model/2"
model = mlflow.sklearn.load_model(model_uri)

@app.post("/predict/")
def predict(transactions: List[TransactionFeatures]):
    # Convert list of pydantic objects to DataFrame
    input_df = pd.DataFrame([t.dict() for t in transactions])
    try:
        predictions = model.predict(input_df)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
