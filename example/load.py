#

import io
import pkg_resources


import pandas as pd
from sklearn.model_selection import train_test_split

from example import settings
from example.persist import cache


###########################################################################
# Loaders
###########################################################################

def load_csv(filename):
    resource = pkg_resources.resource_string(
        "example.resources",
        filename
    )
    return pd.read_csv(io.BytesIO(resource))


###########################################################################
# Files
###########################################################################

def historical_data():
    filename = "historical_data.csv"
    return load_csv(filename)


def predict_data():
    filename = "predict_data.csv"
    return load_csv(filename)


###########################################################################
# Filter
###########################################################################

def filter(frame):
    # NOTE: drop observations that we don't have labels for
    #  not relevant for test data
    mask = ~frame["actual_delivery_time"].isna()
    frame = frame[mask]
    return frame


###########################################################################
# Datasets
###########################################################################

@cache.memoize()
def raw_train(shuffle=False):
    """
    Date range for train:

        min: Timestamp('2014-10-19 05:24:15')
        max: Timestamp('2015-02-18 06:00:44')
    """
    train_dev = historical_data()
    data, _ = train_test_split(
        train_dev,
        shuffle=shuffle,
        train_size=settings.train_size,
        random_state=settings.random
    )
    return filter(data)


@cache.memoize()
def raw_dev(shuffle=False):
    train_dev = historical_data()
    _, data = train_test_split(
        train_dev,
        shuffle=shuffle,
        train_size=settings.train_size,
        random_state=settings.random
    )
    return filter(data)


@cache.memoize()
def raw_test(shuffle=False):
    """
    Date range for train:

        min: Timestamp('2014-10-19 05:24:15')
        max: Timestamp('2015-02-25 05:59:49')
    """
    data = predict_data()
    if shuffle:
        return data.sample(
            frac=1.0, random_state=settings.random)
    return data


###########################################################################
# Main
###########################################################################

def main():
    raw_train()
    raw_dev()
    raw_test()


if __name__ == main():
    main()
