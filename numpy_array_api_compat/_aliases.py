"""
These are functions that are just aliases of existing functions in NumPy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional, Tuple, Union
    from ._typing import ndarray, Device, Dtype, NestedSequence, SupportsBufferProtocol

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

def astype(x: ndarray, dtype: Dtype, /, *, copy: bool = True) -> ndarray:
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

# Creation functions add the device keyword (which does nothing for NumPy)

def _check_device(device):
    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")

# asarray also adds the copy keyword
def asarray(
    obj: Union[
        ndarray,
        bool,
        int,
        float,
        NestedSequence[bool | int | float],
        SupportsBufferProtocol,
    ],
    /,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    copy: Optional[Union[bool, np._CopyMode]] = None,
) -> ndarray:
    _check_device(device)
    if copy in (False, np._CopyMode.IF_NEEDED):
        # copy=False is not yet implemented in np.asarray
        raise NotImplementedError("copy=False is not yet implemented")
    return np.asarray(obj, dtype=dtype)

def arange(
    start: Union[int, float],
    /,
    stop: Optional[Union[int, float]] = None,
    step: Union[int, float] = 1,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> ndarray:
    _check_device(device)
    return np.arange(start, stop=stop, step=step, dtype=dtype)

def empty(
    shape: Union[int, Tuple[int, ...]],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> ndarray:
    _check_device(device)
    return np.empty(shape, dtype=dtype)

def empty_like(
    x: ndarray, /, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None
) -> ndarray:
    _check_device(device)
    return np.empty_like(x, dtype=dtype)

def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    /,
    *,
    k: int = 0,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> ndarray:
    _check_device(device)
    return np.eye(n_rows, M=n_cols, k=k, dtype=dtype)

def full(
    shape: Union[int, Tuple[int, ...]],
    fill_value: Union[int, float],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> ndarray:
    _check_device(device)
    return np.full(shape, fill_value, dtype=dtype)

def full_like(
    x: ndarray,
    /,
    fill_value: Union[int, float],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> ndarray:
    _check_device(device)
    return np.full_like(x, fill_value, dtype=dtype)

def linspace(
    start: Union[int, float],
    stop: Union[int, float],
    /,
    num: int,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    endpoint: bool = True,
) -> ndarray:
    _check_device(device)
    return np.linspace(start, stop, num, dtype=dtype, endpoint=endpoint)

def ones(
    shape: Union[int, Tuple[int, ...]],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> ndarray:
    _check_device(device)
    return np.ones(shape, dtype=dtype)

def ones_like(
    x: ndarray, /, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None
) -> ndarray:
    _check_device(device)
    return np.ones_like(x, dtype=dtype)

def zeros(
    shape: Union[int, Tuple[int, ...]],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> ndarray:
    _check_device(device)
    return np.zeros(shape, dtype=dtype)

def zeros_like(
    x: ndarray, /, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None
) -> ndarray:
    _check_device(device)
    return np.zeros_like(x, dtype=dtype)

# np.reshape calls the keyword argument 'newshape' instead of 'shape'
def reshape(x: ndarray, /, shape: Tuple[int, ...], copy: Optional[bool] = None) -> ndarray:
    if copy is True:
        x = x.copy()
    elif copy is False:
        x.shape = shape
        return x
    return np.reshape(x, shape)

# The descending keyword is new in sort and argsort, and 'kind' replaced with
# 'stable'
def argsort(
    x: ndarray, /, *, axis: int = -1, descending: bool = False, stable: bool = True
) -> ndarray:
    # Note: this keyword argument is different, and the default is different.
    kind = "stable" if stable else "quicksort"
    if not descending:
        res = np.argsort(x, axis=axis, kind=kind)
    else:
        # As NumPy has no native descending sort, we imitate it here. Note that
        # simply flipping the results of np.argsort(x, ...) would not
        # respect the relative order like it would in native descending sorts.
        res = np.flip(
            np.argsort(np.flip(x, axis=axis), axis=axis, kind=kind),
            axis=axis,
        )
        # Rely on flip()/argsort() to validate axis
        normalised_axis = axis if axis >= 0 else x.ndim + axis
        max_i = x.shape[normalised_axis] - 1
        res = max_i - res
    return res

def sort(
    x: ndarray, /, *, axis: int = -1, descending: bool = False, stable: bool = True
) -> ndarray:
    # Note: this keyword argument is different, and the default is different.
    kind = "stable" if stable else "quicksort"
    res = np.sort(x, axis=axis, kind=kind)
    if descending:
        res = np.flip(res, axis=axis)
    return res

# sum() and prod() should always upcast when dtype=None
def sum(
    x: ndarray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype: Optional[Dtype] = None,
    keepdims: bool = False,
) -> ndarray:
    # `np.sum` already upcasts integers, but not floats
    if dtype is None and x.dtype == np.float32:
        dtype = np.float64
    return np.sum(x, axis=axis, dtype=dtype, keepdims=keepdims)

def prod(
    x: ndarray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype: Optional[Dtype] = None,
    keepdims: bool = False,
) -> ndarray:
    if dtype is None and x.dtype == np.float32:
        dtype = np.float64
    return np.prod(x, dtype=dtype, axis=axis, keepdims=keepdims)

# ceil, floor, and trunc return integers for integer inputs

def ceil(x: ndarray, /) -> ndarray:
    if np.issubdtype(x.dtype, np.integer):
        return x
    return np.ceil(x)

def floor(x: ndarray, /) -> ndarray:
    if np.issubdtype(x.dtype, np.integer):
        return x
    return np.floor(x)

def trunc(x: ndarray, /) -> ndarray:
    if np.issubdtype(x.dtype, np.integer):
        return x
    return np.trunc(x)

# from numpy import * doesn't overwrite these builtin names
from numpy import abs, max, min, round

__all__ = ['acos', 'acosh', 'asin', 'asinh', 'atan', 'atan2', 'atanh',
           'bitwise_left_shift', 'bitwise_invert', 'bitwise_right_shift',
           'bool', 'concat', 'pow', 'UniqueAllResult', 'UniqueCountsResult',
           'UniqueInverseResult', 'unique_all', 'unique_counts',
           'unique_inverse', 'unique_values', 'astype', 'abs', 'max', 'min',
           'round', 'std', 'var', 'permute_dims', 'asarray', 'arange',
           'empty', 'empty_like', 'eye', 'full', 'full_like', 'linspace',
           'ones', 'ones_like', 'zeros', 'zeros_like', 'reshape', 'argsort',
           'sort', 'sum', 'prod', 'ceil', 'floor', 'trunc']
