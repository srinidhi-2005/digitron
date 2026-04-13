# model predictions

import pickle
import numpy as np

model = None # global variable

def load_model():
    global model
    if model is None:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
    return model

def predict(image):
    model = load_model()
    output = model.forward(image)
    return int(np.argmax(output, axis=1)[0])

