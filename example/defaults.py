#

import numpy as np
import pandas as pd

from example import settings
from example.persist import cache
from example import preprocess


###########################################################################
# Defaults
###########################################################################

@cache.memoize()
def defaults():
    frame = preprocess.train()
    result = {}
    for col in frame:
        dtype = settings.types[col]
        if dtype == np.dtype("unicode"):
            # NOTE: "-1" is oov for all object columns
            result[col] = "-1"
        elif dtype == np.dtype("int32"):
            # NOTE: -1 is oov for all integer columns
            result[col] = -1
        elif dtype == np.dtype("float32"):
            # NOTE: works because fillna median is idempotent
            mask = ~frame[col].isna()
            result[col] = frame.loc[mask, col].median()
    return result


###########################################################################
# Main
###########################################################################

def main():
    defaults()


if __name__ == "__main__":
    main()
