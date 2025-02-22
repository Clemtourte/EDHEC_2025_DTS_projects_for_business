from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

model = pickle.load(open("model.pkl", "rb"))

class InputPayload(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class OutputPayload(BaseModel):
    prediction: str

@app.post("/infos")
def send_back_information(input: InputPayload):
    return {"message": input}

@app.post("/predict")
def predict_info(input: InputPayload):
    input = input.dict()
    prediction = model.predict([[input["sepal_length"], input["sepal_width"], input["petal_length"], input["petal_width"]]])
    map_predict_to_label = {0: "setosa", 1: "versicolor", 2: "virginica"}
    return {"prediction": map_predict_to_label[prediction[0]]}