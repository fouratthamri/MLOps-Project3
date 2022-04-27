"""
Check Score procedure
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import load
import logging
import numpy as np
from sklearn.metrics import classification_report
from src.train_step import encode_data, process_data


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

def main():
    """
    Execute score checking
    """
    data = pd.read_csv("data/prepared/cleaned_census.csv")
    _, test = train_test_split(data, test_size=0.20)

    #load data pre-processors
    trained_model = load("data/model/random_forest.joblib")
    encoder = load("data/model/columnTransformer.joblib")
    lb = load("data/model/label_binarizer.pkl")
    scaler = load("data/model/scaler.pkl")

    slice_values = []

    for category in cat_features:
        for cls in test[category].unique():
            df_temp = test[test[category] == cls]

            df_temp = encode_data(df_temp, cat_features, 'salary', columnTransformer=encoder, training=False)
            X_test, y_test, _, _ = process_data(df_temp, 'salary', lb=lb, scaler=scaler, training=False)

            y_preds = trained_model.predict(X_test)

            metrics = classification_report(y_test, y_preds)

            line = category + "\n" + cls + "\n" + metrics

            logging.info(line)
            slice_values.append(line)

    with open('data/model/slice_output.txt', 'w') as out:
        for slice_value in slice_values:
            out.write(slice_value + '\n')
