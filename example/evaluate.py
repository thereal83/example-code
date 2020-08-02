#

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from example import settings
from example import preprocess
from example import offline
from example import objective
from example import train
from example import predict


###########################################################################
# Evaluation
###########################################################################

def evaluate_baseline(train_test_dev="dev"):
    y_true, y_pred = predict.predict_baseline("dev")

    score_unweighted = objective.root_mean_squared_error(y_true, y_pred)
    score_weighted = objective.root_mean_squared_error(y_true, y_pred, weight_pos=2.0)

    return score_unweighted, score_weighted


def evaluate_xgboost(train_dev_test="dev", weighted=True):
    _, y_true = offline.build("dev")
    y_pred = predict.predict_xgboost("dev", weighted=weighted)

    score_unweighted = objective.root_mean_squared_error(y_true, y_pred)
    score_weighted = objective.root_mean_squared_error(y_true, y_pred, weight_pos=2.0)

    return score_unweighted, score_weighted


def evaluate_random_forest(train_dev_test="dev"):
    _, y_true = offline.build("dev")
    y_pred = predict.predict_random_forest("dev")

    score_unweighted = objective.root_mean_squared_error(y_true, y_pred)
    score_weighted = objective.root_mean_squared_error(y_true, y_pred, weight_pos=2.0)

    return score_unweighted, score_weighted


def evaluate_pytorch(train_dev_test="dev"):
    raise NotImplementedError


###########################################################################
# Feature importances
###########################################################################

def importances_xgboost(weighted=True):
    bst = train.train_xgboost(weighted=weighted)
    importances = bst.get_score(importance_type="gain")

    results = {}
    for i, (col, val) in enumerate(sorted(importances.items(), key=lambda x: x[1], reverse=True)):
        results[col] = (i, val)
        print(f"{col}: {i} {val}")

    return results


def importances_random_forest():
    features, _ = offline.build()

    reg = train.train_random_forest()
    importances = reg.feature_importances_
    indices = np.argsort(importances)[::-1]

    results = {}
    for i in range(features.shape[1]):
        column = features.columns[i]
        results[column] = (indices[i], importances[indices[i]])
        print(f"{column}: {indices[i]} {importances[indices[i]]}")

    return results


###########################################################################
# Outliers
###########################################################################

def outliers_xgboost():
    """Compare distributions of predictions in our test data
    with distribution of labels in the training data.  The
    percentiles match pretty well but the outliers and std
    are a bit off.
    """
    y_true = np.exp(preprocess.train()[settings.label])
    y_pred = np.exp(predict.predict_xgboost())

    print(y_true.describe())
    print(y_pred.describe())


###########################################################################
# Main
###########################################################################

def main():
    evaluate_xgboost()
    evaluate_random_forest()
    importances_random_forest()
    importances_xgboost()
    outliers_xgboost()


if __name__ == "__main__":
    main()
