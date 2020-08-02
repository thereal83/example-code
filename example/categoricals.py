#

import pandas as pd
import numpy as np

from example.persist import cache
from example import preprocess


###########################################################################
# One Hot
###########################################################################

def make_one_hots(frame, column, allowed_values=None):
    default = preprocess.defaults()[column]
    if allowed_values is None:
        allowed_values = frame[column].unique().tolist()
        uniques = list(set(allowed_values + [default]))
    else:
        uniques = list(set(allowed_values + [default]))

    cats = {}
    for i, val in enumerate(sorted(uniques)):
        one_hot = [0] * len(uniques)
        one_hot[i] = 1
        cats[val] = one_hot

    return cats


###########################################################################
# Categoricals
###########################################################################

@cache.memoize()
def order_protocol_cats():
    frame = preprocess.train()
    return make_one_hots(frame, "order_protocol")


@cache.memoize()
def market_id_cats():
    frame = preprocess.train()
    return make_one_hots(frame, "market_id")


@cache.memoize()
def order_place_estimate_cats():
    frame = preprocess.train()
    return make_one_hots(
        frame,
        "estimated_order_place_duration",
        allowed_values=[251, 446, 0]
    )


###########################################################################
# Main
###########################################################################

def main():
    market_id_cats()
    order_protocol_cats()
    order_place_estimate_cats()


if __name__ == "__main__":
    main()
