from __future__ import annotations

from ...common import _aliases

from ..._internal import get_xp

from ._info import __array_namespace_info__

import numpy as np
from numpy import (
    # Dtypes
    iinfo,
    finfo,
    bool_ as bool,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    complex64,
    complex128,
    can_cast,
    result_type,
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional, Union

    from ...common._typing import Device, Dtype, Array, NestedSequence, SupportsBufferProtocol

import dask.array as da

isdtype = get_xp(np)(_aliases.isdtype)
unstack = get_xp(da)(_aliases.unstack)

# da.astype doesn't respect copy=True
def astype(
    x: Array,
    dtype: Dtype,
    /,
    *,
    copy: bool = True,
    device: Optional[Device] = None
) -> Array:
    """
    Array API compatibility wrapper for astype().

    See the corresponding documentation in the array library and/or the array API
    specification for more details.
    """
    # TODO: respect device keyword?

    if not copy and dtype == x.dtype:
        return x
    x = x.astype(dtype)
    return x.copy() if copy else x

# Common aliases

# This arange func is modified from the common one to
# not pass stop/step as keyword arguments, which will cause
# an error with dask
def arange(
    start: Union[int, float],
    /,
    stop: Optional[Union[int, float]] = None,
    step: Union[int, float] = 1,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    **kwargs,
) -> Array:
    """
    Array API compatibility wrapper for arange().

    See the corresponding documentation in the array library and/or the array API
    specification for more details.
    """
    # TODO: respect device keyword?

    args = [start]
    if stop is not None:
        args.append(stop)
    else:
        # stop is None, so start is actually stop
        # prepend the default value for start which is 0
        args.insert(0, 0)
    args.append(step)

    return da.arange(*args, dtype=dtype, **kwargs)


eye = get_xp(da)(_aliases.eye)
linspace = get_xp(da)(_aliases.linspace)
UniqueAllResult = get_xp(da)(_aliases.UniqueAllResult)
UniqueCountsResult = get_xp(da)(_aliases.UniqueCountsResult)
UniqueInverseResult = get_xp(da)(_aliases.UniqueInverseResult)
unique_all = get_xp(da)(_aliases.unique_all)
unique_counts = get_xp(da)(_aliases.unique_counts)
unique_inverse = get_xp(da)(_aliases.unique_inverse)
unique_values = get_xp(da)(_aliases.unique_values)
permute_dims = get_xp(da)(_aliases.permute_dims)
std = get_xp(da)(_aliases.std)
var = get_xp(da)(_aliases.var)
cumulative_sum = get_xp(da)(_aliases.cumulative_sum)
empty = get_xp(da)(_aliases.empty)
empty_like = get_xp(da)(_aliases.empty_like)
full = get_xp(da)(_aliases.full)
full_like = get_xp(da)(_aliases.full_like)
ones = get_xp(da)(_aliases.ones)
ones_like = get_xp(da)(_aliases.ones_like)
zeros = get_xp(da)(_aliases.zeros)
zeros_like = get_xp(da)(_aliases.zeros_like)
reshape = get_xp(da)(_aliases.reshape)
matrix_transpose = get_xp(da)(_aliases.matrix_transpose)
vecdot = get_xp(da)(_aliases.vecdot)
nonzero = get_xp(da)(_aliases.nonzero)
ceil = get_xp(np)(_aliases.ceil)
floor = get_xp(np)(_aliases.floor)
trunc = get_xp(np)(_aliases.trunc)
matmul = get_xp(np)(_aliases.matmul)
tensordot = get_xp(np)(_aliases.tensordot)
sign = get_xp(np)(_aliases.sign)


# asarray also adds the copy keyword, which is not present in numpy 1.0.
def asarray(
    obj: Union[
        Array,
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
    **kwargs,
) -> Array:
    """
    Array API compatibility wrapper for asarray().

    See the corresponding documentation in the array library and/or the array API
    specification for more details.
    """
    # TODO: respect device keyword?

    if isinstance(obj, da.Array):
        if dtype is not None and dtype != obj.dtype:
            if copy is False:
                raise ValueError("Unable to avoid copy when changing dtype")
            obj = obj.astype(dtype)
        return obj.copy() if copy else obj

    if copy is False:
        raise NotImplementedError(
            "Unable to avoid copy when converting a non-dask object to dask"
        )

    # copy=None to be uniform across dask < 2024.12 and >= 2024.12
    # see https://github.com/dask/dask/pull/11524/
    obj = np.array(obj, dtype=dtype, copy=True)
    return da.from_array(obj)


from dask.array import (
    # Element wise aliases
    arccos as acos,
    arccosh as acosh,
    arcsin as asin,
    arcsinh as asinh,
    arctan as atan,
    arctan2 as atan2,
    arctanh as atanh,
    left_shift as bitwise_left_shift,
    right_shift as bitwise_right_shift,
    invert as bitwise_invert,
    power as pow,
    # Other
    concatenate as concat,
)

# dask.array.clip does not work unless all three arguments are provided.
# Furthermore, the masking workaround in common._aliases.clip cannot work with
# dask (meaning uint64 promoting to float64 is going to just be unfixed for
# now).
def clip(
    x: Array,
    /,
    min: Optional[Union[int, float, Array]] = None,
    max: Optional[Union[int, float, Array]] = None,
) -> Array:
    """
    Array API compatibility wrapper for clip().

    See the corresponding documentation in the array library and/or the array API
    specification for more details.
    """
    def _isscalar(a):
        return isinstance(a, (int, float, type(None)))
    min_shape = () if _isscalar(min) else min.shape
    max_shape = () if _isscalar(max) else max.shape

    # TODO: This won't handle dask unknown shapes
    result_shape = np.broadcast_shapes(x.shape, min_shape, max_shape)

    if min is not None:
        min = da.broadcast_to(da.asarray(min), result_shape)
    if max is not None:
        max = da.broadcast_to(da.asarray(max), result_shape)

    if min is None and max is None:
        return da.positive(x)

    if min is None:
        return astype(da.minimum(x, max), x.dtype)
    if max is None:
        return astype(da.maximum(x, min), x.dtype)

    return astype(da.minimum(da.maximum(x, min), max), x.dtype)

# exclude these from all since dask.array has no sorting functions
_da_unsupported = ['sort', 'argsort']

_common_aliases = [alias for alias in _aliases.__all__ if alias not in _da_unsupported]

__all__ = _common_aliases + ['__array_namespace_info__', 'asarray', 'acos',
                    'acosh', 'asin', 'asinh', 'atan', 'atan2',
                    'atanh', 'bitwise_left_shift', 'bitwise_invert',
                    'bitwise_right_shift', 'concat', 'pow', 'iinfo', 'finfo', 'can_cast',
                    'result_type', 'bool', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64',
                    'uint8', 'uint16', 'uint32', 'uint64',
                    'complex64', 'complex128', 'iinfo', 'finfo',
                    'can_cast', 'result_type']

_all_ignore = ["get_xp", "da", "np"]
