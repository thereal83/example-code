#

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

from example import settings
from example.persist import cache
from example import preprocess


###########################################################################
# Pivot
###########################################################################

def pivot(frame, index):
    tables = []
    for column in settings.lookup_columns:
        table = pd.pivot_table(
            frame,
            values=settings.lookup_values,
            index=index,
            columns=column,
            aggfunc=np.mean,
            fill_value=0.0
        )
        table.columns = [
            index + "_" + str(i)
            for i in range(len(frame[column].unique()))]
        tables.append(table)
    return pd.concat(tables, axis=1).sort_index()


def compress(frame):
    index = frame.index
    column = "_".join(frame.columns[0].split("_")[:-1])
    pipeline = make_pipeline(
        StandardScaler(),
        PCA(
            n_components=settings.n_pca_components,
            whiten=True
        )
    )
    table = pipeline.fit_transform(frame)
    columns = [
        column + "_" + str(i)
        for i in range(settings.n_pca_components)
    ]
    frame = pd.DataFrame(table, index=index, columns=columns)
    return frame


def make_lookup(frame, column):
    lookup = pivot(frame, column)
    # TODO: ensure index contains unique values to lookup
    lookup = compress(lookup)

    # NOTE: dict: index -> list (row.tolist())
    #  dicts are faster to work with than DataFrames
    return lookup.T.to_dict(orient='list')


###########################################################################
# Main
###########################################################################

@cache.memoize()
def store_primary_lookup():
    frame = preprocess.train()
    return make_lookup(frame, "store_primary_category")


@cache.memoize()
def store_id_lookup():
    frame = preprocess.train()
    return make_lookup(frame, "store_id")


###########################################################################
# Main
###########################################################################

def main():
    store_primary_lookup()
    store_id_lookup()


if __name__ == "__main__":
    main()
