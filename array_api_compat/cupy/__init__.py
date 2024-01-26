from cupy import *  # noqa: F401, F403

# from cupy import * doesn't overwrite these builtin names
from cupy import abs, max, min, round

from ..common._helpers import (
    array_namespace,
    device,
    get_namespace,
    is_array_api_obj,
    size,
    to_device,
)

# These imports may overwrite names from the import * above.
from ._aliases import (
    acos,
    acosh,
    asarray,
    asarray_cupy,
    asin,
    asinh,
    atan,
    atan2,
    atanh,
    bitwise_invert,
    bitwise_left_shift,
    bitwise_right_shift,
    bool,
    concat,
    pow,
)
from .linalg import matrix_transpose, vecdot

__all__ = []

__all__ += [
    "abs",
    "max",
    "min",
    "round",
]

__all__ += [
    "is_array_api_obj",
    "array_namespace",
    "get_namespace",
    "device",
    "to_device",
    "size",
]

__all__ += [
    "acos",
    "acosh",
    "asarray",
    "asarray_cupy",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bitwise_invert",
    "bitwise_left_shift",
    "bitwise_right_shift",
    "bool",
    "concat",
    "pow",
]

__all__ += [
    "matrix_transpose",
    "vecdot",
]

# See the comment in the numpy __init__.py
__import__(__package__ + ".linalg")

__array_api_version__ = "2022.12"
