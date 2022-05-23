import os

import numpy as np
import pandas as pd
from fastapi import FastAPI
from joblib import load
from pandas.core.frame import DataFrame
from pydantic import BaseModel, Field

from src.common_functions import get_categorical_features, infer, process_data


class User(BaseModel):
    age: int = Field(..., example=47)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=70878)
    education: str = Field(..., example="Masters")
    education_num: int = Field(..., example=11, alias="education-num")
    marital_status: str = Field(..., example="Separated", alias="marital-status")
    occupation: str = Field(..., example="Sales")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="Asian-Pac-Islander")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=1289, alias="capital-gain")
    capital_loss: int = Field(..., example=50, alias="capital-loss")
    hours_per_week: int = Field(..., example=45, alias="hours-per-week")
    native_country: str = Field(..., example="United-States")


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()


@app.get("/")
async def get_items():
    return {"message": "Greetings!"}


@app.post("/")
async def inference(payload: User):
    model = load("data/model/model.joblib")
    encoder = load("data/model/encoder.joblib")
    lb = load("data/model/lb.joblib")

    print(encoder)

    array = np.array([[
                     payload.age,
                     payload.workclass,
                     payload.fnlgt,
                     payload.education,
                     payload.education_num,
                     payload.marital_status,
                     payload.occupation,
                     payload.relationship,
                     payload.race,
                     payload.sex,
                     payload.capital_gain,
                     payload.capital_loss,
                     payload.hours_per_week,
                     payload.native_country
                     ]])
    data_input = DataFrame(data=array, columns=[
        "age",
        "workclass",
        "fnlgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-loss",
        "capital-gain",
        "hours-per-week",
        "native-country",
    ])

    X, _, _, _ = process_data(
        data_input,
        categorical_features=get_categorical_features(),
        encoder=encoder, lb=lb, training=False)
    prediction = infer(model, X)
    y = lb.inverse_transform(prediction)[0]
    return {"prediction": y}
