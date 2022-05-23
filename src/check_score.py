"""
Evaluate performance of trained model
"""
import logging

import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split

from src.common_functions import (calculate_metrics, get_categorical_features,
                                  process_data)


def main():
    """
    Run performance check
    """
    data = pd.read_csv("data/prepared/census.csv")
    _, test = train_test_split(data, test_size=0.20)

    model = load("data/model/model.joblib")
    encoder = load("data/model/encoder.joblib")
    lb = load("data/model/lb.joblib")

    slice_metrics = []

    for category in get_categorical_features():
        for class_ in test[category].value_counts().index:
            df_temp = test[test[category] == class_]

            X_test, y_test, _, _ = process_data(
                df_temp,
                categorical_features=get_categorical_features(),
                label="salary", encoder=encoder, lb=lb, training=False)

            y_preds = model.predict(X_test)

            precision, recall, f1 = calculate_metrics(y_test, y_preds)

            line = "[%s->%s] Precision: %s " "Recall: %s F1 Score: %s" % (category, class_, precision, recall, f1)
            logging.info(line)
            slice_metrics.append(line)

    with open('data/model/slice_output.txt', 'w') as output:
        for slice_value in slice_metrics:
            output.write(slice_value + '\n')
