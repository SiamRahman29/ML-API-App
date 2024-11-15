from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas

app = FastAPI()

class ScoringItem(BaseModel):
    YearsAtCompany: float
    EmployeeSatisfaction: float 
    Position: str
    Salary: int

with open('rf_model.pkl', 'rb') as file:
    model = pickle.load(file)


@app.post('/')
async def scoring_endpoint(item: ScoringItem):
    df = pandas.DataFrame([item.values()])
    yhat = model.predict(df)
    return {"prediction": int(yhat)}