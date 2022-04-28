"""
Dataset cleaning step
"""
import pandas as pd


def clean_dataset(data):
    """
    Clean the dataset doing some stuff got from eda
    """
    data.replace({'?': None}, inplace=True)
    data.dropna(inplace=True)
    data.drop("fnlgt", axis="columns", inplace=True)
    data.drop("education-num", axis="columns", inplace=True)
    data.drop("capital-gain", axis="columns", inplace=True)
    data.drop("capital-loss", axis="columns", inplace=True)
    return data


def main():
    """
    Run data cleaning process
    """
    df = pd.read_csv("data/raw/census.csv", skipinitialspace=True)
    df = clean_dataset(df)
    df.to_csv("data/prepared/census.csv", index=False)
