# Script to train machine learning model.
from json.tool import main
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn import metrics

from pickle import dump

import pandas as pd
import numpy as np
# Optional enhancement, use K-fold cross validation instead of a train-test split.

def encode_data(data, cat_features, label, columnTransformer=None,  training=True):
    if training:
        columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(sparse=False), cat_features)], remainder='passthrough')
        data = np.array(columnTransformer.fit_transform(data))
        joblib.dump(columnTransformer, 'data/model/columnTransformer.joblib')
    else:
        data = np.array(columnTransformer.transform(data))

    data= pd.DataFrame(data)
    mapping = {data.columns[-1]: label}
    data = data.rename(columns=mapping)
    return data

def process_data(data, label, lb=None, scaler=None, training=True):
    data=data.drop_duplicates(keep="first")

    y_data = data[label]
    data.drop(columns=[label], inplace=True)

    if training:
        scaler = StandardScaler()
        data = pd.DataFrame(scaler.fit_transform(data))

        lb = LabelBinarizer()
        y_data = lb.fit_transform(y_data)
        dump(scaler, open('data/model/scaler.pkl', 'wb'))
        dump(lb, open('data/model/label_binarizer.pkl', 'wb'))

    else:
        data = pd.DataFrame(scaler.transform(data))
        y_data = lb.transform(y_data)

    return data, y_data, scaler, lb

def train_rf(X_train, y_train, n_estimators=100):
    random_forest = RandomForestClassifier(n_estimators=n_estimators)
    random_forest.fit(X_train, y_train)
    joblib.dump(random_forest, "data/model/random_forest.joblib")
    return random_forest

def main():
    # Add code to load in the data.
    data = pd.read_csv('data/prepared/cleaned_census.csv')

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

    data = encode_data(data, cat_features, label='salary')

    train, test = train_test_split(data, test_size=0.20)

    X_train, y_train, _, _ = process_data(
        train, label="salary", training=True
    )
    # Process the test data with the process_data function.

    X_test, y_test, _, _ = process_data(
        test, label="salary", training=True
    )

    model = train_rf(X_train, y_train, n_estimators=100)
    y_pred = model.predict(X_test)

    print(metrics.classification_report(y_test,y_pred))
