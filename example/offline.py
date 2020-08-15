#

import pandas as pd
import numpy as np

from example import settings
from example.persist import cache
from example import defaults
from example import preprocess
from example import categoricals
from example import lookups


###########################################################################
# Featurize
###########################################################################

_categoricals = {
    "market_id": categoricals.market_id_cats,
    "order_protocol": categoricals.order_protocol_cats,
    "estimated_order_place_duration": categoricals.order_place_estimate_cats,
}


def categorical(series, column, build_input_layer=True):
    default = defaults.defaults()[column]
    lookup = pd.DataFrame.from_dict(_categoricals[column]()).T
    columns = [
        "_".join([column, str(i)]) for i in lookup.columns
    ]
    lookup.columns = columns
    mask = series.isin(lookup.index)
    series.where(mask, other=default, inplace=True)

    if build_input_layer:
        return series.to_frame() \
              .merge(
                  lookup,
                  how="left",
                  right_index=True,
                  left_on=column,
              ) \
              .drop(columns=[column]) \
              .reset_index(drop=True)
    else:
        return series.values


_lookups = {
    "store_primary_category": lookups.store_primary_lookup,
    "store_id": lookups.store_id_lookup,
}


def lookup(series, column, build_input_layer=True):
    default = defaults.defaults()[column]
    lookup = pd.DataFrame.from_dict(_lookups[column]()).T
    columns = [
        "_".join([column, str(i)]) for i in lookup.columns
    ]
    lookup.columns = columns
    mask = series.isin(lookup.index)
    series.where(mask, other=default, inplace=True)

    if build_input_layer:
        return series.to_frame() \
              .merge(
                  lookup,
                  how="left",
                  right_index=True,
                  left_on=column,
              ) \
              .drop(columns=[column]) \
              .reset_index(drop=True)
    else:
        series.values


def numeric(series, columnm, build_input_layer=True):
    # TODO (@ebecker): could try adding some featurization
    #   or normalization here
    if build_input_layer:
        return series.to_frame().reset_index(drop=True)
    else:
        return series.values


def featurize(series, dtype, build_input_layer=True):
    if dtype == np.dtype("float32"):
        feature = numeric(series, series.name, build_input_layer)
    elif dtype == np.dtype("int32"):
        feature = categorical(series, series.name, build_input_layer)
    elif dtype == np.dtype("unicode"):
        feature = lookup(series, series.name, build_input_layer)
    else:
        raise ValueError("Unrecognized dtype")
    return feature.astype(np.dtype("float32"))


###########################################################################
# Builders
###########################################################################

@cache.memoize()
def build(train_dev_test="train", build_input_layer=True):
    if train_dev_test == "train":
        frame = preprocess.train()
    elif train_dev_test == "dev":
        frame = preprocess.dev()
    elif train_dev_test == "test":
        frame = preprocess.test()
    else:
         raise ValueError("Unrecognized value for train_dev_test.")

    features = []
    for column in settings.features:
        dtype = settings.types[column]
        series = frame[column]
        feature = featurize(series, dtype, build_input_layer)
        features.append(feature)

    if build_input_layer:
        features = pd.concat(features, axis=1)
    else:
        features = np.concatenate(features, axis=1)

    labels = None
    if train_dev_test != "test":
        column = settings.label
        series = frame[column]
        dtype = settings.types[column]
        labels = featurize(series, dtype, build_input_layer)
        labels = np.squeeze(labels)

    return features, labels


def batch(train_dev_test="train"):
    # TODO (@ebecker): for training with PyTorch
    raise NotImplementedError
