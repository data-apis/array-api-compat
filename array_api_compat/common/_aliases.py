"""
These are functions that are just aliases of existing functions in NumPy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional, Tuple, Union, List
    from ._typing import ndarray, Device, Dtype, NestedSequence, SupportsBufferProtocol

from typing import NamedTuple
from types import ModuleType

from ._helpers import _check_device, _is_numpy_array, get_namespace

# Basic renames
def acos(x, /, xp):
    return xp.arccos(x)

def acosh(x, /, xp):
    return xp.arccosh(x)

def asin(x, /, xp):
    return xp.arcsin(x)

def asinh(x, /, xp):
    return xp.arcsinh(x)

def atan(x, /, xp):
    return xp.arctan(x)

def atan2(x1, x2, /, xp):
    return xp.arctan2(x1, x2)

def atanh(x, /, xp):
    return xp.arctanh(x)

def bitwise_left_shift(x1, x2, /, xp):
    return xp.left_shift(x1, x2)

def bitwise_invert(x, /, xp):
    return xp.invert(x)

def bitwise_right_shift(x1, x2, /, xp):
    return xp.right_shift(x1, x2)

def concat(arrays: Union[Tuple[ndarray, ...], List[ndarray]], /, xp, *, axis: Optional[int] = 0) -> ndarray:
    return xp.concatenate(arrays, axis=axis)

def pow(x1, x2, /, xp):
    return xp.power(x1, x2)

# These functions are modified from the NumPy versions.

def arange(
    start: Union[int, float],
    /,
    stop: Optional[Union[int, float]] = None,
    step: Union[int, float] = 1,
    *,
    xp,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> ndarray:
    _check_device(xp, device)
    return xp.arange(start, stop=stop, step=step, dtype=dtype)

def empty(
    shape: Union[int, Tuple[int, ...]],
    xp,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> ndarray:
    _check_device(xp, device)
    return xp.empty(shape, dtype=dtype)

def empty_like(
    x: ndarray, /, xp, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None
) -> ndarray:
    _check_device(xp, device)
    return xp.empty_like(x, dtype=dtype)

def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    /,
    *,
    xp,
    k: int = 0,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> ndarray:
    _check_device(xp, device)
    return xp.eye(n_rows, M=n_cols, k=k, dtype=dtype)

def full(
    shape: Union[int, Tuple[int, ...]],
    fill_value: Union[int, float],
    xp,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> ndarray:
    _check_device(xp, device)
    return xp.full(shape, fill_value, dtype=dtype)

def full_like(
    x: ndarray,
    /,
    fill_value: Union[int, float],
    *,
    xp,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> ndarray:
    _check_device(xp, device)
    return xp.full_like(x, fill_value, dtype=dtype)

def linspace(
    start: Union[int, float],
    stop: Union[int, float],
    /,
    num: int,
    *,
    xp,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    endpoint: bool = True,
) -> ndarray:
    _check_device(xp, device)
    return xp.linspace(start, stop, num, dtype=dtype, endpoint=endpoint)

def ones(
    shape: Union[int, Tuple[int, ...]],
    xp,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> ndarray:
    _check_device(xp, device)
    return xp.ones(shape, dtype=dtype)

def ones_like(
    x: ndarray, /, xp, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None
) -> ndarray:
    _check_device(xp, device)
    return xp.ones_like(x, dtype=dtype)

def zeros(
    shape: Union[int, Tuple[int, ...]],
    xp,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> ndarray:
    _check_device(xp, device)
    return xp.zeros(shape, dtype=dtype)

def zeros_like(
    x: ndarray, /, xp, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None
) -> ndarray:
    _check_device(xp, device)
    return xp.zeros_like(x, dtype=dtype)

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


def unique_all(x: ndarray, /, xp) -> UniqueAllResult:
    values, indices, inverse_indices, counts = xp.unique(
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


def unique_counts(x: ndarray, /, xp) -> UniqueCountsResult:
    res = xp.unique(
        x,
        return_counts=True,
        return_index=False,
        return_inverse=False,
        equal_nan=False,
    )

    return UniqueCountsResult(*res)


def unique_inverse(x: ndarray, /, xp) -> UniqueInverseResult:
    values, inverse_indices = xp.unique(
        x,
        return_counts=False,
        return_index=False,
        return_inverse=True,
        equal_nan=False,
    )
    # xp.unique() flattens inverse indices, but they need to share x's shape
    # See https://github.com/numpy/numpy/issues/20638
    inverse_indices = inverse_indices.reshape(x.shape)
    return UniqueInverseResult(values, inverse_indices)


def unique_values(x: ndarray, /, xp) -> ndarray:
    return xp.unique(
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
    xp,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    correction: Union[int, float] = 0.0, # correction instead of ddof
    keepdims: bool = False,
) -> ndarray:
    return xp.std(x, axis=axis, ddof=correction, keepdims=keepdims)

def var(
    x: ndarray,
    /,
    xp,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    correction: Union[int, float] = 0.0, # correction instead of ddof
    keepdims: bool = False,
) -> ndarray:
    return xp.var(x, axis=axis, ddof=correction, keepdims=keepdims)

# Unlike transpose(), the axes argument to permute_dims() is required.
def permute_dims(x: ndarray, /, axes: Tuple[int, ...], xp) -> ndarray:
    return xp.transpose(x, axes)

# Creation functions add the device keyword (which does nothing for NumPy)

# asarray also adds the copy keyword
def _asarray(
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
    copy: "Optional[Union[bool, np._CopyMode]]" = None,
    namespace = None,
) -> ndarray:
    """
    Array API compatibility wrapper for asarray().

    See the corresponding documentation in NumPy/CuPy and/or the array API
    specification for more details.

    """
    if namespace is None:
        try:
            xp = get_namespace(obj, _use_compat=False)
        except ValueError:
            # TODO: What about lists of arrays?
            raise ValueError("A namespace must be specified for asarray() with non-array input")
    elif isinstance(namespace, ModuleType):
        xp = namespace
    elif namespace == 'numpy':
        import numpy as xp
    elif namespace == 'cupy':
        import cupy as xp
    else:
        raise ValueError("Unrecognized namespace argument to asarray()")

    _check_device(xp, device)
    if _is_numpy_array(obj):
        import numpy as np
        COPY_FALSE = (False, np._CopyMode.IF_NEEDED)
        COPY_TRUE = (True, np._CopyMode.ALWAYS)
    else:
        COPY_FALSE = (False,)
        COPY_TRUE = (True,)
    if copy in COPY_FALSE:
        # copy=False is not yet implemented in xp.asarray
        raise NotImplementedError("copy=False is not yet implemented")
    if isinstance(obj, xp.ndarray):
        if dtype is not None and obj.dtype != dtype:
            copy = True
        if copy in COPY_TRUE:
            return xp.array(obj, copy=True, dtype=dtype)
        return obj

    return xp.asarray(obj, dtype=dtype)

# xp.reshape calls the keyword argument 'newshape' instead of 'shape'
def reshape(x: ndarray, /, shape: Tuple[int, ...], xp, copy: Optional[bool] = None) -> ndarray:
    if copy is True:
        x = x.copy()
    elif copy is False:
        x.shape = shape
        return x
    return xp.reshape(x, shape)

# The descending keyword is new in sort and argsort, and 'kind' replaced with
# 'stable'
def argsort(
    x: ndarray, /, xp, *, axis: int = -1, descending: bool = False, stable: bool = True
) -> ndarray:
    # Note: this keyword argument is different, and the default is different.
    kind = "stable" if stable else "quicksort"
    if not descending:
        res = xp.argsort(x, axis=axis, kind=kind)
    else:
        # As NumPy has no native descending sort, we imitate it here. Note that
        # simply flipping the results of xp.argsort(x, ...) would not
        # respect the relative order like it would in native descending sorts.
        res = xp.flip(
            xp.argsort(xp.flip(x, axis=axis), axis=axis, kind=kind),
            axis=axis,
        )
        # Rely on flip()/argsort() to validate axis
        normalised_axis = axis if axis >= 0 else x.ndim + axis
        max_i = x.shape[normalised_axis] - 1
        res = max_i - res
    return res

def sort(
    x: ndarray, /, xp, *, axis: int = -1, descending: bool = False, stable: bool = True
) -> ndarray:
    # Note: this keyword argument is different, and the default is different.
    kind = "stable" if stable else "quicksort"
    res = xp.sort(x, axis=axis, kind=kind)
    if descending:
        res = xp.flip(res, axis=axis)
    return res

# sum() and prod() should always upcast when dtype=None
def sum(
    x: ndarray,
    /,
    xp,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype: Optional[Dtype] = None,
    keepdims: bool = False,
) -> ndarray:
    # `xp.sum` already upcasts integers, but not floats
    if dtype is None and x.dtype == xp.float32:
        dtype = xp.float64
    return xp.sum(x, axis=axis, dtype=dtype, keepdims=keepdims)

def prod(
    x: ndarray,
    /,
    xp,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype: Optional[Dtype] = None,
    keepdims: bool = False,
) -> ndarray:
    if dtype is None and x.dtype == xp.float32:
        dtype = xp.float64
    return xp.prod(x, dtype=dtype, axis=axis, keepdims=keepdims)

# ceil, floor, and trunc return integers for integer inputs

def ceil(x: ndarray, /, xp) -> ndarray:
    if xp.issubdtype(x.dtype, xp.integer):
        return x
    return xp.ceil(x)

def floor(x: ndarray, /, xp) -> ndarray:
    if xp.issubdtype(x.dtype, xp.integer):
        return x
    return xp.floor(x)

def trunc(x: ndarray, /, xp) -> ndarray:
    if xp.issubdtype(x.dtype, xp.integer):
        return x
    return xp.trunc(x)

__all__ = ['acos', 'acosh', 'asin', 'asinh', 'atan', 'atan2', 'atanh',
           'bitwise_left_shift', 'bitwise_invert', 'bitwise_right_shift',
           'concat', 'pow', 'UniqueAllResult', 'UniqueCountsResult',
           'UniqueInverseResult', 'unique_all', 'unique_counts',
           'unique_inverse', 'unique_values', 'astype', 'std', 'var',
           'permute_dims', 'reshape', 'argsort', 'sort', 'sum', 'prod',
           'ceil', 'floor', 'trunc']
