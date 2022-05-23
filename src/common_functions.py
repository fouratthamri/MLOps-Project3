"""
Main functions module
"""


import logging

import numpy as np
from numpy import mean, std
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def get_categorical_features():
    """ Return feature categories
    """
    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    return categorical_features


def process_data(
    X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])
    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb



def train_rf(X_train, y_train):
    """
    Trains a random forest model and returns it
    Inputs
    ------
    X_train : np.array
        Training data array.
    y_train : np.array
        Targets array
    Outputs
    -------
    model
        Trained machine learning model.
    """
    cv_splits = KFold(n_splits=7, shuffle=True)
    model = RandomForestClassifier(n_estimators=50)
    model.fit(X_train, y_train)
    scores = cross_val_score(model, X_train, y_train, scoring='accuracy',
                             cv=cv_splits, n_jobs=-1)
    logging.info('Accuracy (mean and std): %.5f (%.5f)' % (mean(scores), std(scores)))
    return model


def calculate_metrics(y, preds):
    """
    Calculates the trained model classification metrics: Precision, Recall and F1 score

    Inputs
    ------
    y : np.array
        True labels.
    preds : np.array
        Predicted labels.
    Outputs
    -------
    precision : np.float
    recall : np.float
    fbeta : np.float
    """
    precision, recall, f1 = precision_score(y, preds), recall_score(y, preds), fbeta_score(y, preds, beta=1), 
    return precision, recall, f1


def infer(model, X):
    """ Run model inference and output predictions.

    Inputs
    ------
    model : sklearn model object
        Trained random forest model.
    X : np.array
        Inference input data.
    Returns
    -------
    preds : np.array
        Model predictions.
    """
    preds = model.predict(X)
    return preds
