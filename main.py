from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model
model = joblib.load("electricity_model.pkl")

app = FastAPI(title="Electricity Cost Prediction API")

# Input schema
class InputData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: InputData):
    try:
        prediction = model.predict([np.array(data.features)])
        return {"Predicted Electricity Cost": prediction[0]}
    except Exception as e:
        return {"error": str(e)}
