#

import numpy as np
import pandas as pd

from example import settings
from example.persist import cache
from example import load


###########################################################################
# Cast
###########################################################################

def cast_raw(frame):
    for col, dtype in settings.raw_types.items():
        if col in frame:
            frame[col] = frame[col].astype(dtype)
    return frame


def cast(frame):
    for col in settings.types:
        dtype = settings.types[col]
        if col in frame:
            frame[col] = frame[col].astype(dtype)
    return frame


###########################################################################
# Impute
###########################################################################

@cache.memoize()
def defaults():
    # TODO (@ebecker): can improve on these defaults
    frame = load.raw_train()
    result = {}
    for col in frame:
        dtype = settings.raw_types[col]
        if dtype == np.dtype("unicode"):
            # NOTE: "-1" is oov for all object columns
            result[col] = "-1"
        elif dtype == np.dtype("int32"):
            # NOTE: -1 is oov for all integer columns
            result[col] = -1
        elif dtype == np.dtype("float32"):
            mask = ~frame[col].isna()
            result[col] = frame.loc[mask, col].median()
    return result


def fillna(frame):
    for col in frame:
        if col not in defaults():
            continue
        mask = frame[col].isna()
        frame.loc[mask, col] = defaults()[col]
    return frame


###########################################################################
# Augment columns
###########################################################################

def augment(frame, train_dev_test="train"):
    # frame["created_at_woy"] = frame["created_at"].dt.isocalendar().week
    frame["created_at_dow"] = frame["created_at"].dt.isocalendar().day
    frame["created_at_hod"] = frame["created_at"].dt.hour
    frame["created_at_min"] = frame["created_at"].dt.minute

    if train_dev_test != "test":
        series = frame["actual_delivery_time"] - frame["created_at"]

        frame["delivery_duration"] = series.dt.seconds
        # NOTE: transform targets because they look to be log-normally
        #   distrbuted.  Optionally, in xgboost, can use a logloss
        #   objective to accomplish a similar result
        frame["delivery_duration"] = np.log(frame["delivery_duration"])

    return frame


###########################################################################
# Mask
###########################################################################

def mask(frame, train_dev_test="train"):
    """I mask stores in the training data because there are lots
    of OOV store_ids in the test data.
    """
    if train_dev_test != "train":
        return frame

    mask = np.full(len(frame), False)
    n = int(len(frame) * settings.mask_size)
    mask[:n] = True
    settings.random.shuffle(mask)
    frame.loc[mask, "store_id"] = defaults()["store_id"]
    return frame


###########################################################################
# Preprocess
###########################################################################

def preprocess(frame, train_dev_test="train"):
    frame = fillna(frame)
    frame = cast_raw(frame)
    frame = augment(frame, train_dev_test)
    frame = mask(frame, train_dev_test)
    frame = cast(frame)
    return frame


###########################################################################
# Datasets
###########################################################################

@cache.memoize()
def train():
    return preprocess(load.raw_train(), "train")


@cache.memoize()
def dev():
    return preprocess(load.raw_dev(), "dev")


@cache.memoize()
def test():
    return preprocess(load.raw_test(), "test")


###########################################################################
# Main
###########################################################################

def main():
    train()
    dev()
    test()


if __name__ == "__main__":
    main()
