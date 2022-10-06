"""
These are functions that are just aliases of existing functions in NumPy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional, Tuple, Union
    from numpy import ndarray, dtype

from typing import NamedTuple

import numpy as np

# Basic renames
acos = np.arccos
acosh = np.arccosh
asin = np.arcsin
asinh = np.arcsinh
atan = np.arctan
atan2 = np.arctan2
atanh = np.arctanh
bitwise_left_shift = np.left_shift
bitwise_invert = np.invert
bitwise_right_shift = np.right_shift
bool = np.bool_
concat = np.concatenate
pow = np.power

# These functions are modified from the NumPy versions.

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
    values, indices, inverse_indices, counts = np.unique(
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
    res = np.unique(
        x,
        return_counts=True,
        return_index=False,
        return_inverse=False,
        equal_nan=False,
    )

    return UniqueCountsResult(*res)


def unique_inverse(x: ndarray, /) -> UniqueInverseResult:
    values, inverse_indices = np.unique(
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
    return np.unique(
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

# These functions have different keyword argument names

def std(
    x: ndarray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    correction: Union[int, float] = 0.0, # correction instead of ddof
    keepdims: bool = False,
) -> ndarray:
    return np.std(x, axis=axis, ddof=correction, keepdims=keepdims)

def var(
    x: ndarray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    correction: Union[int, float] = 0.0, # correction instead of ddof
    keepdims: bool = False,
) -> ndarray:
    return np.var(x, axis=axis, ddof=correction, keepdims=keepdims)

# Unlike transpose(), the axes argument to permute_dims() is required.
def permute_dims(x: ndarray, /, axes: Tuple[int, ...]) -> ndarray:
    return np.transpose(x, axes)

# from numpy import * doesn't overwrite these builtin names
from numpy import abs, max, min, round

__all__ = ['acos', 'acosh', 'asin', 'asinh', 'atan', 'atan2', 'atanh',
           'bitwise_left_shift', 'bitwise_invert', 'bitwise_right_shift',
           'bool', 'concat', 'pow', 'UniqueAllResult', 'UniqueCountsResult',
           'UniqueInverseResult', 'unique_all', 'unique_counts',
           'unique_inverse', 'unique_values', 'astype', 'abs', 'max', 'min',
           'round', 'std', 'var', 'permute_dims']
