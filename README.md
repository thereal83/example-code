Example Code
=======================

This repository trains a machine-learned model to predict delivery times for
orders on a hypothetical meal delivery service.

The model uses a variety of features to make predictions including: the
time and day of the order, store id, features of the items in the order,
market, order processing time, drive time between store and customer,
and dasher supply and demand.  I designed this model for offline prediction
and can easily adapt the codebase to support real-time inference.


Quickstart
----------

How to install this package and run the code.

1. Install the package (I highly recommend installing
inside a virtual environment):

    ```bash
    pip install -e .[dev]
    ```

    **WARNING**: `pip install` may create a new directory on your system.
    See the section on caching to learn more.


2. Run the code:

    ```bash
    python -u example/cicd.py
    ```

Learn more by inspecting the codebase.  Running `cicd.py` will skip over
some of the analysis.  For example, it won't generate a submission
file.


Evaluations
-----

The following table presents best runs from various models.  rmsle stands
for root mean squared error where the error is a difference in logs.  a-rmsle
is an assymetric version of rmsle such that the loss associated with late
orders is 2x the loss associated with early orders.


| model                   |  rmsle  | a-rmsle |
|-------------------------|---------|---------|
| baseline                | 0.52942 | 0.68893 |
| random forest           | 0.30224 | 0.37565 |
| xgboost                 | 0.28777 | 0.35845 |
| xgboost w / custom loss | 0.29666 | 0.34627 |


I performed hyperparemeter tuning for each model so these results
represent the best run over that search space.


Caching
----

This repository uses the package `DiskCache` for on-disc caching.  The cache is a
key-value store where keys reflect the signature of your function call.  You can
list keys in the cache with:

``` python
from example.persist import cache
[key for key in cache.iterkeys]
```

The first time you call a function decorated with `@cache.memoize` the result
is saved to disc.  Subsequent calls returned a cached copy of the result without
re-running the function.  The cache persists between Python sessions.  If you
want to re-run a function (due to an up-stream code change, say) you must
first clear the the cache.  The following command clears everything
from the cache:

``` python
from example.persist import cache
cache.clear()
```

The following command clears a single key-value pair from the cache:

``` python
from example.persist import cache
cache.pop(<key>)
```

Complete documentation for `DisKCache` is available
[here](http://www.grantjenks.com/docs/diskcache/api.html).

The storage location for the on-disc cache is system-specific and is
determined dynamically at install time.  Run the following command to
print the cache directory on your system:

``` python
from example import persist
print(persist.cache_dir)
```

**WARNING**: `pip` will create this directory (if it doesn't already exist)
when you install this pacakge.  Read more about `appdirs`
[here](https://github.com/ActiveState/appdirs).
