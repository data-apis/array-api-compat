from numpy import *  # noqa: F401, F403
from numpy import __all__ as _numpy_all

# from numpy import * doesn't overwrite these builtin names
from numpy import abs, max, min, round

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
    asarray_numpy,
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
    vecdot,
    zeros,
    zeros_like,
)

__all__ = []

__all__ += _numpy_all

__all__ += [
    "abs",
    "max",
    "min",
    "round",
]

__all__ += [
    "array_namespace",
    "device",
    "get_namespace",
    "is_array_api_obj",
    "size",
    "to_device",
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
    "asarray_numpy",
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

# Don't know why, but we have to do an absolute import to import linalg. If we
# instead do
#
# from . import linalg
#
# It doesn't overwrite np.linalg from above. The import is generated
# dynamically so that the library can be vendored.
__import__(__package__ + ".linalg")

__array_api_version__ = "2022.12"
