"""
Helper functions module
"""
import logging
import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score
from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
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
    df, categorical_features=[], label=None, training=True, encoder=None,
    lb=None
):
    """ Preprocesses data:

    Encode categorical features and Binarizes target labels

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in
        `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will
        be returned
        for y (default=None)
    training : bool
        check if in inference or training mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Categorical Encoder
    lb : sklearn.preprocessing._label.LabelBinarizer
        Label Binarizer

    Outputs
    -------
    X : np.array
        Input features dataset
    y : np.array
        Target labels, None if in inference mode
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        OneHotEncoder
    lb : sklearn.preprocessing._label.LabelBinarizer
        LabelBinarizer
    """

    if label is not None:
        y = df[label]
        df = df.drop([label], axis=1)
    else:
        y = np.array([])

    df_categorical = df[categorical_features].values
    df_continuous = df.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        df_categorical = encoder.fit_transform(df_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        df_categorical = encoder.transform(df_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([df_continuous, df_categorical], axis=1)
    return X, y, encoder, lb


def train_rf(X_train, y_train):
    """
    Trains a random forest model and returns it
    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Outputs
    -------
    model
        Trained machine learning model.
    """
    cv = KFold(n_splits=10, shuffle=True, random_state=1)
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X_train, y_train)
    scores = cross_val_score(model, X_train, y_train, scoring='accuracy',
                             cv=cv, n_jobs=-1)
    logging.info('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall,
    and F1.

    Inputs
    ------
    y : np.array
        True labels.
    preds : np.array
        Predicted labels.
    Outputs
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

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
    y_preds = model.predict(X)
    return y_preds
