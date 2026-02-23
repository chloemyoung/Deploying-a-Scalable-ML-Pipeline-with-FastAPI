import pytest
import numpy as np
import pandas as pd

from ml.model import train_model, inference, compute_model_metrics
from ml.data import process_data

# Small dataset for testing
data = pd.DataFrame({
    "age": [25, 35, 50],
    "workclass": ["Private", "Self-emp", "Government"],
    "education": ["Bachelors", "Masters", "HS-grad"],
    "marital-status": ["Never-married", "Married", "Divorced"],
    "occupation": ["Tech-support", "Exec-managerial", "Adm-clerical"],
    "relationship": ["Not-in-family", "Husband", "Unmarried"],
    "race": ["White", "Black", "Asian-Pac-Islander"],
    "sex": ["Male", "Female", "Male"],
    "hours-per-week": [40, 50, 30],
    "native-country": ["United-States", "United-States", "India"],
    "salary": [">50K", "<=50K", "<=50K"]
})

cat_features = [
    "workclass", "education", "marital-status",
    "occupation", "relationship", "race", "sex", "native-country"
]


# Test 1: train_model returns a trained model

def test_train_model_returns_model():
    """
    Check that train_model returns a non-None model object
    """
    X, y, encoder, lb = process_data(
        X=data, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X, y)
    assert model is not None


# Test 2: inference returns predictions of correct length

def test_inference_length():
    """
    Check that inference returns the same number of predictions as input rows
    """
    X, y, encoder, lb = process_data(
        X=data, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X, y)
    preds = inference(model, X)
    assert len(preds) == X.shape[0]


# Test 3: compute_model_metrics outputs valid metrics

def test_compute_model_metrics_values():
    """
    Check that precision, recall, and f1 are all between 0 and 1
    """
    X, y, encoder, lb = process_data(
        X=data, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X, y)
    preds = inference(model, X)
    p, r, f1 = compute_model_metrics(y, preds)
    assert 0.0 <= p <= 1.0
    assert 0.0 <= r <= 1.0
    assert 0.0 <= f1 <= 1.0
