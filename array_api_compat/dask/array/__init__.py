from typing import Final

from dask.array import *  # noqa: F403

# The above is missing a wealth of stuff
import dask.array as da
__all__ = [n for n in dir(da) if not n.startswith("_")]
globals().update({n: getattr(da, n) for n in __all__})
del da

# These imports may overwrite names from the import * above.
from . import _aliases
from ._aliases import *  # noqa: F403
from ._info import __array_namespace_info__  # noqa: F401

__array_api_version__: Final = "2024.12"
del Final

# See the comment in the numpy __init__.py
__import__(__package__ + '.linalg')
__import__(__package__ + '.fft')

__all__ += _aliases.__all__
__all__ += ["__array_api_version__", "__array_namespace_info__", "linalg", "fft"]
__all__ = sorted(set(__all__))
