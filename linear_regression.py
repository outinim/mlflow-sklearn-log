from pprint import pprint

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import mlflow.sklearn
import pandas as pd

import mlflow
from utils import fetch_logged_data


def main():

    # enable autologging
    mlflow.sklearn.autolog()

    # Read the wine-quality csv file from the URL
    csv_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    data = pd.read_csv(csv_url, sep=";")

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # train a model
    model = linear_model.ElasticNet(alpha=0.1)
    with mlflow.start_run() as run:

        # Log artifacts: columns used for modeling
        cols_x = pd.DataFrame(list(train_x.columns))
        cols_x.to_csv("features.csv", header=False, index=False)
        mlflow.log_artifact("features.csv")

        cols_y = pd.DataFrame(list(train_y.columns))
        cols_y.to_csv("targets.csv", header=False, index=False)
        mlflow.log_artifact("targets.csv")

        model.fit(train_x, train_y)
        print("Logged data and model in run {}".format(run.info.run_id))

    # show logged data
    for key, data in fetch_logged_data(run.info.run_id).items():
        print("\n---------- logged {} ----------".format(key))
        pprint(data)

    mlflow.sklearn.log_model(model, "model")


if __name__ == "__main__":
    main()
