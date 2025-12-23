from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Churn Tahmin Servisi", description="...")

with open("best_churn_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

class CustomerEntry(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
    TotalServices: int
    AverageCostPerService: float
    new_customer: int
    Is_auto_payment: int

@app.post("/predict")
def predict(data: CustomerEntry):
    df_input = pd.DataFrame([data.dict()])
    prediction = loaded_model.predict(df_input)[0]
    probability = loaded_model.predict_proba(df_input)[0][1]
    return {"prediction": int(prediction), "probability": float(probability)}

@app.get("/")
def home():
    return {"message": "Live!"}
    