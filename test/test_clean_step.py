"""
Basic cleaning module test
"""
import pandas as pd
import pytest
import src.clean_step


@pytest.fixture
def data():
    """
    Get dataset
    """
    df = src.clean_step.clean_data()
    return df


def test_null(data):
    """
    Data is assumed to have no null values
    """
    assert data.shape == data.dropna().shape


def test_remove_cols(data):
    """
    Check if irrelevant cols were removed
    """
    assert "fnlgt" not in data.columns
    assert "education-num" not in data.columns
    assert "capital-gain" not in data.columns
    assert "capital-loss" not in data.columns

def test_wrong_vals(data):
    """
    Data is assumed to have no question marks value
    """
    assert '?' not in data.values



