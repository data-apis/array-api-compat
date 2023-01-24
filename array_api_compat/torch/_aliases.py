from __future__ import annotations

from functools import wraps
from builtins import all as builtin_all

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import List, Optional, Tuple, Union
    from ..common._typing import Device
    from torch import dtype as Dtype

import torch
array = torch.Tensor

_array_api_dtypes = {
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float32,
    torch.float64,
}

_promotion_table  = {
    # bool
    (torch.bool, torch.bool): torch.bool,
    # ints
    (torch.int8, torch.int8): torch.int8,
    (torch.int8, torch.int16): torch.int16,
    (torch.int8, torch.int32): torch.int32,
    (torch.int8, torch.int64): torch.int64,
    (torch.int16, torch.int8): torch.int16,
    (torch.int16, torch.int16): torch.int16,
    (torch.int16, torch.int32): torch.int32,
    (torch.int16, torch.int64): torch.int64,
    (torch.int32, torch.int8): torch.int32,
    (torch.int32, torch.int16): torch.int32,
    (torch.int32, torch.int32): torch.int32,
    (torch.int32, torch.int64): torch.int64,
    (torch.int64, torch.int8): torch.int64,
    (torch.int64, torch.int16): torch.int64,
    (torch.int64, torch.int32): torch.int64,
    (torch.int64, torch.int64): torch.int64,
    # uints
    (torch.uint8, torch.uint8): torch.uint8,
    # ints and uints (mixed sign)
    (torch.int8, torch.uint8): torch.int16,
    (torch.int16, torch.uint8): torch.int16,
    (torch.int32, torch.uint8): torch.int32,
    (torch.int64, torch.uint8): torch.int64,
    (torch.uint8, torch.int8): torch.int16,
    (torch.uint8, torch.int16): torch.int16,
    (torch.uint8, torch.int32): torch.int32,
    (torch.uint8, torch.int64): torch.int64,
    # floats
    (torch.float32, torch.float32): torch.float32,
    (torch.float32, torch.float64): torch.float64,
    (torch.float64, torch.float32): torch.float64,
    (torch.float64, torch.float64): torch.float64,
}


def _two_arg(f):
    @wraps(f)
    def _f(x1, x2, /, **kwargs):
        x1, x2 = _fix_promotion(x1, x2)
        return f(x1, x2, **kwargs)
    if _f.__doc__ is None:
        _f.__doc__ = f"""\
Array API compatibility wrapper for torch.{f.__name__}.

See the corresponding PyTorch documentation and/or the array API specification
for more details.

"""
    return _f

def _fix_promotion(x1, x2):
    if x1.dtype not in _array_api_dtypes or x2.dtype not in _array_api_dtypes:
        return x1, x2
    # If an argument is 0-D pytorch downcasts the other argument
    if x1.shape == ():
        dtype = result_type(x1, x2)
        x2 = x2.to(dtype)
    if x2.shape == ():
        dtype = result_type(x1, x2)
        x1 = x1.to(dtype)
    return x1, x2

def result_type(*arrays_and_dtypes: Union[array, Dtype]) -> Dtype:
    if len(arrays_and_dtypes) == 0:
        raise TypeError("At least one array or dtype must be provided")
    if len(arrays_and_dtypes) == 1:
        x = arrays_and_dtypes[0]
        if isinstance(x, torch.dtype):
            return x
        return x.dtype
    if len(arrays_and_dtypes) > 2:
        return result_type(arrays_and_dtypes[0], result_type(*arrays_and_dtypes[1:]))

    x, y = arrays_and_dtypes
    xdt = x.dtype if not isinstance(x, torch.dtype) else x
    ydt = y.dtype if not isinstance(y, torch.dtype) else y

    if (xdt, ydt) in _promotion_table:
        return _promotion_table[xdt, ydt]

    # This doesn't result_type(dtype, dtype) for non-array API dtypes
    # because torch.result_type only accepts tensors. This does however, allow
    # cross-kind promotion.
    return torch.result_type(x, y)

def can_cast(from_: Union[dtype, array], to: Dtype, /) -> bool:
    if not isinstance(from_, torch.dtype):
        from_ = from_.dtype
    return torch.can_cast(from_, to)

# Basic renames
permute_dims = torch.permute
bitwise_invert = torch.bitwise_not

# Two-arg elementwise functions
# These require a wrapper to do the correct type promotion on 0-D tensors
add = _two_arg(torch.add)
atan2 = _two_arg(torch.atan2)
bitwise_and = _two_arg(torch.bitwise_and)
bitwise_left_shift = _two_arg(torch.bitwise_left_shift)
bitwise_or = _two_arg(torch.bitwise_or)
bitwise_right_shift = _two_arg(torch.bitwise_right_shift)
bitwise_xor = _two_arg(torch.bitwise_xor)
divide = _two_arg(torch.divide)
# Also a rename. torch.equal does not broadcast
equal = _two_arg(torch.eq)
floor_divide = _two_arg(torch.floor_divide)
greater = _two_arg(torch.greater)
greater_equal = _two_arg(torch.greater_equal)
less = _two_arg(torch.less)
less_equal = _two_arg(torch.less_equal)
logaddexp = _two_arg(torch.logaddexp)
# logical functions are not included here because they only accept bool in the
# spec, so type promotion is irrelevant.
multiply = _two_arg(torch.multiply)
not_equal = _two_arg(torch.not_equal)
pow = _two_arg(torch.pow)
remainder = _two_arg(torch.remainder)
subtract = _two_arg(torch.subtract)

# These wrappers are mostly based on the fact that pytorch uses 'dim' instead
# of 'axis'.

# torch.min and torch.max return a tuple and don't support multiple axes https://github.com/pytorch/pytorch/issues/58745
def max(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> array:
    # https://github.com/pytorch/pytorch/issues/29137
    if axis == ():
        return torch.clone(x)
    return torch.amax(x, axis, keepdims=keepdims)

def min(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> array:
    # https://github.com/pytorch/pytorch/issues/29137
    if axis == ():
        return torch.clone(x)
    return torch.amin(x, axis, keepdims=keepdims)

def _normalize_axes(axis, ndim):
    axes = []
    lower, upper = -ndim, ndim - 1
    for a in axis:
        if a < lower or a > upper:
            # Match torch error message (e.g., from sum())
            raise IndexError(f"Dimension out of range (expected to be in range of [{lower}, {upper}], but got {a}")
        if a < 0:
            a = a + ndim
        if a in axes:
            # Match torch error message but use IndexError instead of RuntimeError
            raise IndexError(f"dim {a} appears multiple times in the list of dims")
        axes.append(a)
    return sorted(axes)


def _apply_keepdims(x, ndim, keepdims):
    if keepdims:
        return x[(None,)*ndim]
    return x

def prod(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, dtype: Optional[Dtype] = None, keepdims: bool = False) -> array:
    # torch.prod doesn't support multiple axes
    # (https://github.com/pytorch/pytorch/issues/56586).
    x = torch.asarray(x)
    ndim = x.ndim
    if isinstance(axis, tuple):
        axes = _normalize_axes(axis, x.ndim)
        for i, a in enumerate(axes):
            if keepdims:
                x = torch.prod(x, a, dtype=dtype)
                x = torch.unsqueeze(x, a)
            else:
                x = torch.prod(x, a - i, dtype=dtype)
        return x
    if axis is None:
        # torch doesn't support keepdims with axis=None
        # (https://github.com/pytorch/pytorch/issues/71209)
        res = torch.prod(x, dtype=dtype)
        res = _apply_keepdims(res, ndim, keepdims)
        return res

    return torch.prod(x, axis, dtype=dtype, keepdims=keepdims)

def any(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> array:
    # torch.any doesn't support multiple axes
    # (https://github.com/pytorch/pytorch/issues/56586).
    x = torch.asarray(x)
    ndim = x.ndim
    if axis == ():
        return x.to(torch.bool)
    if isinstance(axis, tuple):
        axes = _normalize_axes(axis, x.ndim)
        for i, a in enumerate(axes):
            if keepdims:
                x = torch.any(x, a)
                x = torch.unsqueeze(x, a)
            else:
                x = torch.any(x, a - i)
        return x.to(torch.bool)
    if axis is None:
        # torch doesn't support keepdims with axis=None
        # (https://github.com/pytorch/pytorch/issues/71209)
        res = torch.any(x)
        res = _apply_keepdims(res, ndim, keepdims)
        return res.to(torch.bool)

    # torch.any doesn't return bool for uint8
    return torch.any(x, axis, keepdims=keepdims).to(torch.bool)

def all(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> array:
    # torch.all doesn't support multiple axes
    # (https://github.com/pytorch/pytorch/issues/56586).
    x = torch.asarray(x)
    ndim = x.ndim
    if axis == ():
        return x.to(torch.bool)
    if isinstance(axis, tuple):
        axes = _normalize_axes(axis, x.ndim)
        for i, a in enumerate(axes):
            if keepdims:
                x = torch.all(x, a)
                x = torch.unsqueeze(x, a)
            else:
                x = torch.all(x, a - i)
        return x.to(torch.bool)
    if axis is None:
        # torch doesn't support keepdims with axis=None
        # (https://github.com/pytorch/pytorch/issues/71209)
        res = torch.all(x)
        res = _apply_keepdims(res, ndim, keepdims)
        return res.to(torch.bool)

    # torch.all doesn't return bool for uint8
    return torch.all(x, axis, keepdims=keepdims).to(torch.bool)

# torch.arange doesn't support returning empty arrays
# (https://github.com/pytorch/pytorch/issues/70915), and doesn't support some
# keyword argument combinations
# (https://github.com/pytorch/pytorch/issues/70914)
def arange(start: Union[int, float],
           /,
           stop: Optional[Union[int, float]] = None,
           step: Union[int, float] = 1,
           *,
           dtype: Optional[Dtype] = None,
           device: Optional[Device] = None,
           **kwargs) -> array:
    if stop is None:
        start, stop = 0, start
    if step > 0 and stop <= start or step < 0 and stop >= start:
        if dtype is None:
            if builtin_all(isinstance(i, int) for i in [start, stop, step]):
                dtype = torch.int64
            else:
                dtype = torch.float32
        return torch.empty(0, dtype=dtype, device=device, **kwargs)
    return torch.arange(start, stop, step, dtype=dtype, device=device, **kwargs)

# torch.eye does not accept None as a default for the second argument and
# doesn't support off-diagonals (https://github.com/pytorch/pytorch/issues/70910)
def eye(n_rows: int,
        n_cols: Optional[int] = None,
        /,
        *,
        k: int = 0,
        dtype: Optional[Dtype] = None,
        device: Optional[Device] = None,
        **kwargs) -> array:
    if n_cols is None:
        n_cols = n_rows
    z = torch.zeros(n_rows, n_cols, dtype=dtype, device=device, **kwargs)
    if abs(k) <= n_rows + n_cols:
        z.diagonal(k).fill_(1)
    return z

# torch.linspace doesn't have the endpoint parameter
def linspace(start: Union[int, float],
             stop: Union[int, float],
             /,
             num: int,
             *,
             dtype: Optional[Dtype] = None,
             device: Optional[Device] = None,
             endpoint: bool = True,
             **kwargs) -> array:
    if not endpoint:
        return torch.linspace(start, stop, num+1, dtype=dtype, device=device, **kwargs)[:-1]
    return torch.linspace(start, stop, num, dtype=dtype, device=device, **kwargs)

# torch.full does not accept an int size
# https://github.com/pytorch/pytorch/issues/70906
def full(shape: Union[int, Tuple[int, ...]],
         fill_value: Union[bool, int, float, complex],
         *,
         dtype: Optional[Dtype] = None,
         device: Optional[Device] = None,
         **kwargs) -> array:
    if isinstance(shape, int):
        shape = (shape,)

    return torch.full(shape, fill_value, dtype=dtype, device=device, **kwargs)

# Functions that aren't in torch https://github.com/pytorch/pytorch/issues/58742
def expand_dims(x: array, /, *, axis: int = 0) -> array:
    return torch.unsqueeze(x, axis)

def astype(x: array, dtype: Dtype, /, *, copy: bool = True) -> array:
    return x.to(dtype, copy=copy)

def broadcast_arrays(*arrays: array) -> List[array]:
    shape = torch.broadcast_shapes(*[a.shape for a in arrays])
    return [torch.broadcast_to(a, shape) for a in arrays]

__all__ = ['result_type', 'can_cast', 'permute_dims', 'bitwise_invert', 'add',
           'atan2', 'bitwise_and', 'bitwise_left_shift', 'bitwise_or',
           'bitwise_right_shift', 'bitwise_xor', 'divide', 'equal',
           'floor_divide', 'greater', 'greater_equal', 'less', 'less_equal',
           'logaddexp', 'multiply', 'not_equal', 'pow', 'remainder',
           'subtract', 'max', 'min', 'prod', 'any', 'all', 'arange', 'eye',
           'linspace', 'full', 'expand_dims', 'astype', 'broadcast_arrays']
