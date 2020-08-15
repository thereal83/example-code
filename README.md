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

Eager to get started? Follow these steps:

1. Install the package (I highly recommend installing
inside a virtual environment):

    ```bash
    pip install -e .[dev]
    ```

    **WARNING**: running this command may create a new directory
    on your system.  See the section on caching to learn more.


2. Run the code:

    ```bash
    python -u example/cicd.py
    ```

Learn more by inspecting the codebase.


Caching
----

This repository uses the package `DiskCache` for on-disc caching.

``` python
from turo import persist
from turo.persist import cache
```

The cache is a key-value store where keys reflect the signature of your function
call.  You can list keys in the cache with:

``` python
[key for key in cache.iterkeys]
```

This project uses the DiskCache `memoize` decorator.  The first time you call a
function decorated with `@cache.memoize` the result is saved to disc.
Subsequent calls returned a cached copy of the previous result.

The cache persists between Python sessions.  If you
want to re-run a function (due to an up-stream code change, say) you must
first clear the the cache.  To clear everything from the cache, run:

``` python
cache.clear()
```

To clear a single key-value pair from the cache, run:

``` python
cache.pop(<key>)
```

The on-disc cache is saved to a system-specific directory that we
determine dynamically at install time using the AppDirs Python package.
The following command prints the cache directory on your system:

``` python
print(persist.cache_dir)
```

**WARNING**: `setuptools` will create this directory (if it doesn't
already exist) when you install this pacakge.  Uninstalling this package
will not remove the directory.

Read more about `appdirs`
[here](https://github.com/ActiveState/appdirs).

Complete documentation for `DisKCache` is available
[here](http://www.grantjenks.com/docs/diskcache/api.html).
