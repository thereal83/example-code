#

import pandas as pd
import numpy as np
import xgboost as xgb

from example import settings
from example import preprocess
from example import offline
from example import train


###########################################################################
# Predict
###########################################################################

def predict_baseline(train_dev_test="test"):
    pd.set_option('use_inf_as_na', True)
    if train_dev_test == "train":
        frame = preprocess.train()
    elif train_dev_test == "dev":
        frame = preprocess.dev()
    else:
         raise ValueError("Unrecognized value for train_dev_test.")

    model = train.train_baseline()
    frame = frame.merge(
        model,
        left_on="store_primary_category",
        right_index=True,
        how="left"
    )

    frame = frame[~frame["remaining"].isna()]
    y_pred = (
        np.log(frame["estimated_store_to_consumer_driving_duration"])
        + frame["remaining"]
    )
    y_true = frame[settings.label]

    return y_true, y_pred


def predict_xgboost(train_dev_test="test", weighted=True):
    features, _ = offline.build(train_dev_test)
    ddata = xgb.DMatrix(features)

    bst = train.train_xgboost(weighted=weighted)
    y_pred = bst.predict(ddata)

    y_pred = pd.Series(y_pred, index=features.index)
    return y_pred


def predict_random_forest(train_dev_test="test"):
    features, _ = offline.build(train_dev_test)

    regr = train.train_random_forest()
    y_pred = regr.predict(features)

    y_pred = pd.Series(y_pred, index=features.index)
    return y_pred


def predict_pytorch(train_dev_test="test"):
    raise NotImplementedError


###########################################################################
# Submission
###########################################################################

def assemble_submission():
    test = preprocess.test()
    y_pred = predict_xgboost(weighted=True)
    y_pred = np.exp(y_pred).round().astype(np.dtype("int32"))
    frame = pd.concat([test["delivery_id"], y_pred], axis=1)
    frame.columns = ["delivery_id", "predicted_duration"]
    frame.reset_index(drop=True, inplace=True)
    return frame


###########################################################################
# Main
###########################################################################

def main():
    predict_xgboost(weighted=True)
    predict_xgboost(weighted=False)
    predict_random_forest()
    assemble_submission()
    # train_pytorch()


if __name__ == "__main__":
    main()
