#

import io
import json
import appdirs
from diskcache import Cache

from example import __appname__

cache_dir = appdirs.user_data_dir(__appname__)
cache = Cache(cache_dir)
