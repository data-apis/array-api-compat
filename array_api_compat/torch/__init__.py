# Several names are not included in the above import *
import torch
from torch import *  # noqa: F401, F403

from .._internal import _get_all_public_members


def filter_(name):
    if (
        name.startswith("_")
        or name.endswith("_")
        or "cuda" in name
        or "cpu" in name
        or "backward" in name
    ):
        return False
    return True


_torch_all = _get_all_public_members(torch, filter_=filter_)

for _name in _torch_all:
    globals()[_name] = getattr(torch, _name)


from ..common._helpers import (  # noqa: E402
    array_namespace,
    device,
    get_namespace,
    is_array_api_obj,
    size,
    to_device,
)

# These imports may overwrite names from the import * above.
from ._aliases import (  # noqa: E402
    add,
    all,
    any,
    arange,
    astype,
    atan2,
    bitwise_and,
    bitwise_invert,
    bitwise_left_shift,
    bitwise_or,
    bitwise_right_shift,
    bitwise_xor,
    broadcast_arrays,
    broadcast_to,
    can_cast,
    concat,
    divide,
    empty,
    equal,
    expand_dims,
    eye,
    flip,
    floor_divide,
    full,
    greater,
    greater_equal,
    isdtype,
    less,
    less_equal,
    linspace,
    logaddexp,
    matmul,
    matrix_transpose,
    max,
    mean,
    min,
    multiply,
    newaxis,
    nonzero,
    not_equal,
    ones,
    permute_dims,
    pow,
    prod,
    remainder,
    reshape,
    result_type,
    roll,
    sort,
    squeeze,
    std,
    subtract,
    sum,
    take,
    tensordot,
    tril,
    triu,
    unique_all,
    unique_counts,
    unique_inverse,
    unique_values,
    var,
    vecdot,
    where,
    zeros,
)

__all__ = []

__all__ += _torch_all

__all__ += [
    "is_array_api_obj",
    "array_namespace",
    "get_namespace",
    "device",
    "to_device",
    "size",
]

__all__ += [
    "result_type",
    "can_cast",
    "permute_dims",
    "bitwise_invert",
    "newaxis",
    "add",
    "atan2",
    "bitwise_and",
    "bitwise_left_shift",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "divide",
    "equal",
    "floor_divide",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "logaddexp",
    "multiply",
    "not_equal",
    "pow",
    "remainder",
    "subtract",
    "max",
    "min",
    "sort",
    "prod",
    "sum",
    "any",
    "all",
    "mean",
    "std",
    "var",
    "concat",
    "squeeze",
    "broadcast_to",
    "flip",
    "roll",
    "nonzero",
    "where",
    "reshape",
    "arange",
    "eye",
    "linspace",
    "full",
    "ones",
    "zeros",
    "empty",
    "tril",
    "triu",
    "expand_dims",
    "astype",
    "broadcast_arrays",
    "unique_all",
    "unique_counts",
    "unique_inverse",
    "unique_values",
    "matmul",
    "matrix_transpose",
    "vecdot",
    "tensordot",
    "isdtype",
    "take",
]


# See the comment in the numpy __init__.py
__import__(__package__ + ".linalg")

__array_api_version__ = "2022.12"
