"""
Common functions module test
"""
import numpy as np
import pandas as pd
import pytest
import sklearn
from joblib import load

from src.common_functions import (calculate_metrics, get_categorical_features,
                                  process_data, train_rf)


@pytest.fixture
def data():
    """
    Load census dataset
    """
    df = pd.read_csv("data/prepared/census.csv")
    return df


def test_process_data(data):
    encoder = load("data/model/encoder.joblib")
    lb = load("data/model/lb.joblib")

    X, y, _, _ = process_data(
        data,
        categorical_features = get_categorical_features(),
        label="salary", encoder=encoder, lb=lb, training=False)

    assert type(X) == np.ndarray
    assert type(y) == np.ndarray
    assert len(X) == len(y)

def test_train_rf(data):
    encoder = load("data/model/encoder.joblib")
    lb = load("data/model/lb.joblib")

    X, y, _, _ = process_data(
        data,
        categorical_features = get_categorical_features(),
        label="salary", encoder=encoder, lb=lb, training=False)

    model = train_rf(X, y)
    assert type(model) == sklearn.ensemble._forest.RandomForestClassifier


def test_calculate_metrics(data):
    encoder = load("data/model/encoder.joblib")
    lb = load("data/model/lb.joblib")

    X, y, _, _ = process_data(
        data,
        categorical_features = get_categorical_features(),
        label="salary", encoder=encoder, lb=lb, training=False)
    model = train_rf(X, y)
    preds = model.predict(X)

    assert len(calculate_metrics(y,preds)) == 3

