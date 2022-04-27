# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Fourat Thamri created the model. It is a Random Forest classifier using the default hyperparameters in scikit-learn.

## Intended Use

This model should be used to predict the salary of a person based on data about its financials.

## Training Data

The traning data comes from https://archive.ics.uci.edu/ml/datasets/census+income (80% of the full data)

## Evaluation Data

The testing data comes also from https://archive.ics.uci.edu/ml/datasets/census+income ; (20% of the full data)

## Metrics

Model evaluation was done using the accuracy metric which is mostly around 0.80

## Ethical Considerations

The dataset contains race, gender and origin country related data. This will result to a model that may be biased toward specific classes.
Careful investigation of the model bias should be done beforehand.

## Caveats and Recommendations


