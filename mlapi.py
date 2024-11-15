from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas

app = FastAPI()

class ScoringItem(BaseModel):
    SepalLengthCm: float
    SepalWidthCm: float
    PetalLengthCm: float
    PetalWidthCm: float

with open('iris_classifier_model.pkl', 'rb') as file:
    model = pickle.load(file)


@app.post('/')
async def scoring_endpoint(item: ScoringItem):
    df = pandas.DataFrame([item.dict().values()])
    yhat = model.predict(df)
    result = ""
    if yhat == 0:
        result = "Iris-setosa"
    elif yhat == 1:
        result = "Iris-versicolor"
    else:
        result = "Iris-virginica"

    return {"prediction": result}