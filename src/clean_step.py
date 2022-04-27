import pandas as pd


def clean_data():
    """
    data cleaning step
    """

    census = pd.read_csv("data/raw/census.csv", skipinitialspace=True)
    census.dropna(inplace=True)

    for feature in census:
        if not census[feature].dtype.kind == "O":
            continue
        if census[feature].str.contains("\?").any():
            census.loc[census[feature].str.contains("\?"), feature] = "Other"

    census.drop("fnlgt", axis="columns", inplace=True)
    census.drop("education-num", axis="columns", inplace=True)
    census.drop("capital-gain", axis="columns", inplace=True)
    census.drop("capital-loss", axis="columns", inplace=True)

    census.to_csv('data/prepared/cleaned_census.csv', index=False)
    return census
