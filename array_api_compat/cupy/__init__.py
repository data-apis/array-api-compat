import cupy as _cp
from cupy import *  # noqa: F401, F403

# from cupy import * doesn't overwrite these builtin names
from cupy import abs, max, min, round

from .._internal import _get_all_public_members
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
    UniqueAllResult,
    UniqueCountsResult,
    UniqueInverseResult,
    acos,
    acosh,
    arange,
    argsort,
    asarray,
    asarray_cupy,
    asin,
    asinh,
    astype,
    atan,
    atan2,
    atanh,
    bitwise_invert,
    bitwise_left_shift,
    bitwise_right_shift,
    bool,
    ceil,
    concat,
    empty,
    empty_like,
    eye,
    floor,
    full,
    full_like,
    isdtype,
    linspace,
    matmul,
    matrix_transpose,
    nonzero,
    ones,
    ones_like,
    permute_dims,
    pow,
    prod,
    reshape,
    sort,
    std,
    sum,
    tensordot,
    trunc,
    unique_all,
    unique_counts,
    unique_inverse,
    unique_values,
    var,
    zeros,
    zeros_like,
)
from .linalg import matrix_transpose, vecdot

__all__ = []

__all__ += _get_all_public_members(_cp)

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
    "UniqueAllResult",
    "UniqueCountsResult",
    "UniqueInverseResult",
    "acos",
    "acosh",
    "arange",
    "argsort",
    "asarray",
    "asarray_cupy",
    "asin",
    "asinh",
    "astype",
    "atan",
    "atan2",
    "atanh",
    "bitwise_invert",
    "bitwise_left_shift",
    "bitwise_right_shift",
    "bool",
    "ceil",
    "concat",
    "empty",
    "empty_like",
    "eye",
    "floor",
    "full",
    "full_like",
    "isdtype",
    "linspace",
    "matmul",
    "matrix_transpose",
    "nonzero",
    "ones",
    "ones_like",
    "permute_dims",
    "pow",
    "prod",
    "reshape",
    "sort",
    "std",
    "sum",
    "tensordot",
    "trunc",
    "unique_all",
    "unique_counts",
    "unique_inverse",
    "unique_values",
    "var",
    "zeros",
    "zeros_like",
]

__all__ += [
    "matrix_transpose",
    "vecdot",
]

# See the comment in the numpy __init__.py
__import__(__package__ + ".linalg")

__array_api_version__ = "2022.12"
