import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model


# Pydantic model for incoming data

class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")


# Load saved encoder and model
encoder_path = os.path.join("model", "encoder.pkl")  # adjust path if needed
model_path = os.path.join("model", "model.pkl")      # adjust path if needed

encoder = load_model(encoder_path)
model = load_model(model_path)


# Create FastAPI app
app = FastAPI()


# GET endpoint at root
@app.get("/")
async def get_root():
    """Say hello"""
    return {"message": "Welcome to the ML inference API!"}


# POST endpoint for inference
@app.post("/data/")
async def post_inference(data: Data):
    # Turn Pydantic model into dict
    data_dict = data.dict()
    # Replace underscores with hyphens for DataFrame
    data_df = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    data_df = pd.DataFrame.from_dict(data_df)

    # Define categorical features
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Process data using existing encoder (training=False)
    data_processed, _, _, _ = process_data(
        X=data_df,
        categorical_features=cat_features,
        label=None,      # no label for inference
        training=False,
        encoder=encoder,
        lb=None          # label binarizer not needed
    )

    # Run inference
    _inference = inference(model, data_processed)

    # Return result using apply_label to convert 0/1 to <=50K/>50K
    return {"result": apply_label(_inference)}
