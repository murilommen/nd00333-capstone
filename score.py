import json
import numpy as np
import os
import joblib
from azureml.core.model import Model 


def init():
    global model
    model_path = Model.get_model_path("HD-heart-disease")
    model = joblib.load(model_path)

def run(data):
    try:
        data = np.array(json.loads(data))
        result = model.predict(data)
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error