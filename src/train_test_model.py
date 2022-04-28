"""
Model trainig module
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump
import src.helper_functions


def main():
    """
    Run model training
    """
    df = pd.read_csv("data/prepared/census.csv")
    train, _ = train_test_split(df, test_size=0.20)

    X_train, y_train, encoder, lb = src.helper_functions.process_data(
        train, categorical_features=src.helper_functions.get_categorical_features(),
        label="salary", training=True
    )

    print(X_train.dtype)
    trained_model = src.helper_functions.train_rf(X_train, y_train)

    dump(trained_model, "data/model/model.joblib")
    dump(encoder, "data/model/encoder.joblib")
    dump(lb, "data/model/lb.joblib")
