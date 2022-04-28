"""
Basic cleaning module test
"""
import pandas as pd
import pytest
import src.basic_cleaning


@pytest.fixture
def data():
    """
    Load Dataset
    """
    df = pd.read_csv("data/raw/census.csv", skipinitialspace=True)
    df = src.basic_cleaning.clean_dataset(df)
    return df


def test_wrong_vals(data):
    """
    Check whether all the eronous rows was removed
    """
    assert '?' not in data.values

def test_missing_vals(data):
    """
    Check if all missing values were removed
    """
    assert data.shape == data.dropna().shape



