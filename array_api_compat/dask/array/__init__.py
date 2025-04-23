from typing import Final

import dask.array as da
from dask.array import *  # noqa: F403

# These imports may overwrite names from the import * above.
from . import _aliases
from ._aliases import *  # noqa: F403
from ._info import __array_namespace_info__  # noqa: F401

__array_api_version__: Final = "2024.12"

# See the comment in the numpy __init__.py
__import__(__package__ + '.linalg')
__import__(__package__ + '.fft')

def _make_all(base):
    return sorted(
        set(base) 
        | set(_aliases.__all__) 
        | {"__array_api_version__", "__array_namespace_info__", "linalg", "fft"}
    )

__all__ = _make_all(da.__all__)

def __dir__() -> list[str]:
    return _make_all(dir(da))
