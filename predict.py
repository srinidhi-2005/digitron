# model predictions

import pickle
import numpy as np

def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()  # load once with fast API

def predict(image):
    output = model.forward(image)
    return int(np.argmax(output, axis=1)[0])

