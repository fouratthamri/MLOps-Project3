"""
Dataset cleaning step
"""

import pandas as pd


def clean_dataset(data):
    """
    Function to remove NaN values
    """
    data.replace({'?': None}, inplace=True)
    data.dropna(inplace=True)
    return data


def main():
    """
    Run data cleaning process
    """
    data = pd.read_csv("data/raw/census.csv")
    data.columns = data.columns.str.replace(' ', '')
    data['salary'] = data['salary'].apply(lambda x: x.replace(' ', ''))
    data = clean_dataset(data)
    data.to_csv("data/prepared/census.csv", index=False)
