"""
These are functions that are just aliases of existing functions in NumPy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Tuple
    from numpy import ndarray, dtype

from typing import NamedTuple

from numpy import (arccos, arccosh, arcsin, arcsinh, arctan, arctan2, arctanh,
                   left_shift, invert, right_shift, bool_, concatenate, power,
                   transpose, unique)

# Basic renames
acos = arccos
acosh = arccosh
asin = arcsin
asinh = arcsinh
atan = arctan
atan2 = arctan2
atanh = arctanh
bitwise_left_shift = left_shift
bitwise_invert = invert
bitwise_right_shift = right_shift
bool = bool_
concat = concatenate
pow = power

# These functions are modified from the NumPy versions.

# Unlike transpose(), the axes argument to permute_dims() is required.
def permute_dims(x: ndarray, /, axes: Tuple[int, ...]) -> ndarray:
    return transpose(x, axes)

# np.unique() is split into four functions in the array API:
# unique_all, unique_counts, unique_inverse, and unique_values (this is done
# to remove polymorphic return types).

# The functions here return namedtuples (np.unique() returns a normal
# tuple).
class UniqueAllResult(NamedTuple):
    values: ndarray
    indices: ndarray
    inverse_indices: ndarray
    counts: ndarray


class UniqueCountsResult(NamedTuple):
    values: ndarray
    counts: ndarray


class UniqueInverseResult(NamedTuple):
    values: ndarray
    inverse_indices: ndarray


def unique_all(x: ndarray, /) -> UniqueAllResult:
    values, indices, inverse_indices, counts = unique(
        x,
        return_counts=True,
        return_index=True,
        return_inverse=True,
        equal_nan=False,
    )
    # np.unique() flattens inverse indices, but they need to share x's shape
    # See https://github.com/numpy/numpy/issues/20638
    inverse_indices = inverse_indices.reshape(x.shape)
    return UniqueAllResult(
        values,
        indices,
        inverse_indices,
        counts,
    )


def unique_counts(x: ndarray, /) -> UniqueCountsResult:
    res = unique(
        x,
        return_counts=True,
        return_index=False,
        return_inverse=False,
        equal_nan=False,
    )

    return UniqueCountsResult(*res)


def unique_inverse(x: ndarray, /) -> UniqueInverseResult:
    values, inverse_indices = unique(
        x,
        return_counts=False,
        return_index=False,
        return_inverse=True,
        equal_nan=False,
    )
    # np.unique() flattens inverse indices, but they need to share x's shape
    # See https://github.com/numpy/numpy/issues/20638
    inverse_indices = inverse_indices.reshape(x.shape)
    return UniqueInverseResult(values, inverse_indices)


def unique_values(x: ndarray, /) -> ndarray:
    return unique(
        x,
        return_counts=False,
        return_index=False,
        return_inverse=False,
        equal_nan=False,
    )

def astype(x: ndarray, dtype: dtype, /, *, copy: bool = True) -> ndarray:
    if not copy and dtype == x.dtype:
        return x
    return x.astype(dtype=dtype, copy=copy)

# from numpy import * doesn't overwrite these builtin names
from numpy import abs, max, min, round
