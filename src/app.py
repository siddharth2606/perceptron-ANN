import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import os 
import numpy as np 
from keras.models import load_model

app = FastAPI()

base_bim = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_bim,"src","perceptron.pkl")
ann_model = os.path.join(base_bim,"src","ANN.h5")
scaler = os.path.join(base_bim,"src","ANN.pkl")

model = joblib.load(model_path)
ann_model = load_model(ann_model)
scaler = joblib.load(scaler)

class predictInput(BaseModel):
    cgpa : float
    score : float

class annInput(BaseModel):
    CreditScore: float
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: float
    HasCrCard: float
    IsActiveMember: float
    EstimatedSalary: float


@app.post("/predict")
def predict(data:predictInput):
    input_data = [[data.cgpa,data.score]]
    prediction = model.predict(input_data)

    return{
        "prediction" : int(prediction[0])
    }

@app.post("/predict-ann")
def predict_churn(data: annInput):
    input_data = np.array([[
        data.CreditScore,
        data.Age,
        data.Tenure,
        data.Balance,
        data.NumOfProducts,
        data.HasCrCard,
        data.IsActiveMember,
        data.EstimatedSalary
    ]])

    input_data = scaler.transform(input_data)
    prediction = ann_model.predict(input_data)

    return {
        "raw_prediction": float(prediction[0][0]),
        "churn": int(prediction[0][0] > 0.5)
    }