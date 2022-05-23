"""
Model training module
"""

import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split

from src.common_functions import (get_categorical_features, process_data,
                                  train_rf)


def main():
    """
    Run model training
    """
    data = pd.read_csv("data/prepared/census.csv")
    train, _ = train_test_split(data, test_size=0.20)
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=get_categorical_features(),
        label="salary", training=True)
    model = train_rf(X_train, y_train)

    dump(model, "data/model/model.joblib")
    dump(encoder, "data/model/encoder.joblib")
    dump(lb, "data/model/lb.joblib")
    exit(0)
