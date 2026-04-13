# import required libraries

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from predict import predict

app = FastAPI()

class InputData(BaseModel):
    input: list

@app.get("/")
def home():
    return {"message": "MNIST API is running"}

@app.post("/predict")
def predict_digit(data: InputData):
    image = np.array(data.input).reshape(1, 784)
    result = predict(image)
    return {"prediction": result}

