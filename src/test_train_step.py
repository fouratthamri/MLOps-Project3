# import pandas as pd
# import numpy as np
# from pandas.core.frame import DataFrame
# import src
# import pytest
# from joblib import load

# cat_features = [
#     "workclass",
#     "education",
#     "marital-status",
#     "occupation",
#     "relationship",
#     "race",
#     "sex",
#     "native-country",
# ]

# def test_process_encoder(data):
#     """
#     Check split have same number of rows for X and y
#     """
#     encoder_test = load("data/model/encoder.joblib")
#     lb_test = load("data/model/lb.joblib")

#     _, _, encoder, lb = src.train_test_step.process_data(
#         data,
#         categorical_features=cat_features,
#         label="salary", training=True)

#     _, _, _, _ = src.train_test_step.process_data(
#         data,
#         categorical_features=cat_features,
#         label="salary", encoder=encoder_test, lb=lb_test, training=False)

#     assert encoder.get_params() == encoder_test.get_params()
#     assert lb.get_params() == lb_test.get_params()