"""
Tests for basic_cleaning module functions
"""

import pandas as pd
import pytest

import src.basic_cleaning


@pytest.fixture
def data():
    """
    Loads data
    """
    data = pd.read_csv("data/raw/census.csv")
    data = src.basic_cleaning.clean_dataset(data)
    return data


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



