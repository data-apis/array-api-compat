from typing import Final

from .._internal import clone_module


__all__ = clone_module("mlx.core", globals())

# These imports may overwrite names from the import * above.
from . import _aliases
from ._aliases import *  # type: ignore[assignment,no-redef] # noqa: F403
from ._info import __array_namespace_info__  # noqa: F401

# Don't know why, but we have to do an absolute import to import linalg. If we
# instead do
#
# from . import linalg
#
# It doesn't overwrite np.linalg from above. The import is generated
# dynamically so that the library can be vendored.
__import__(__spec__.parent + ".linalg")

__import__(__spec__.parent + ".fft")

from .linalg import matrix_transpose, vecdot  # type: ignore[no-redef]  # noqa: F401

__array_api_version__: Final = "2025.12"

__all__ = sorted(
    set(__all__)
    | set(_aliases.__all__)
    | {"__array_api_version__", "__array_namespace_info__", "linalg", "fft"}
)

def __dir__() -> list[str]:
    return __all__


# Monkeypatch mlx.core.array to support boolean indexing __getitem__
# MLX doesn't natively support boolean indexing __getitem__ (data-dependent shapes),
# so we provide a NumPy-based fallback.
import mlx.core as mx
import numpy as np
import builtins

_old_getitem = mx.array.__getitem__

def _is_bool_array(x):
    return isinstance(x, mx.array) and x.dtype == mx.bool_

def _has_bool_array(item):
    if _is_bool_array(item):
        return True
    if isinstance(item, tuple):
        return builtins.any(_has_bool_array(i) for i in item)
    if isinstance(item, list):
        return builtins.any(_has_bool_array(i) for i in item)
    return False

def _to_numpy_indices(item):
    if _is_bool_array(item):
        return np.array(item)
    if isinstance(item, tuple):
        return tuple(_to_numpy_indices(i) for i in item)
    if isinstance(item, list):
        return [_to_numpy_indices(i) for i in item]
    return item

def _new_getitem(self, item):
    if _has_bool_array(item):
        # Convert self and any boolean masks to numpy
        self_np = np.array(self)
        item_np = _to_numpy_indices(item)
        res_np = self_np[item_np]
        return mx.array(res_np)
    return _old_getitem(self, item)

mx.array.__getitem__ = _new_getitem
