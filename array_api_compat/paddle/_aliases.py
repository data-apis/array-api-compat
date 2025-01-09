from __future__ import annotations

from typing import Literal
import numpy as np

from functools import wraps as _wraps
from builtins import any as _builtin_any

from ..common._aliases import (
    unstack as _aliases_unstack,
)
from ..common._typing import (
    SupportsBufferProtocol,
    NestedSequence,
)
from .._internal import get_xp

from ._info import __array_namespace_info__

import paddle

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List, Optional, Sequence, Tuple, Union
    from ..common._typing import Device
    from paddle import dtype as Dtype

    array = paddle.Tensor

_int_dtypes = {
    paddle.uint8,
    paddle.int8,
    paddle.int16,
    paddle.int32,
    paddle.int64,
}

_array_api_dtypes = {
    paddle.bool,
    *_int_dtypes,
    paddle.float32,
    paddle.float64,
    paddle.complex64,
    paddle.complex128,
}

# NOTE: Implicit promotion rules of Paddle is a bit strict than other frameworks,
# see details: https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/advanced/auto_type_promotion_cn.html
_promotion_table = {
    # bool
    (paddle.bool, paddle.bool): paddle.bool,
    # ints
    (paddle.int8, paddle.int8): paddle.int8,
    (paddle.int16, paddle.int16): paddle.int16,
    (paddle.int32, paddle.int32): paddle.int32,
    (paddle.int64, paddle.int64): paddle.int64,
    # uints
    (paddle.uint8, paddle.uint8): paddle.uint8,
    # floats
    (paddle.float32, paddle.float32): paddle.float32,
    (paddle.float32, paddle.float64): paddle.float64,
    (paddle.float64, paddle.float32): paddle.float64,
    (paddle.float64, paddle.float64): paddle.float64,
    # complexes
    (paddle.complex64, paddle.complex64): paddle.complex64,
    (paddle.complex64, paddle.complex128): paddle.complex128,
    (paddle.complex128, paddle.complex64): paddle.complex128,
    (paddle.complex128, paddle.complex128): paddle.complex128,
    # Mixed float and complex
    (paddle.float32, paddle.complex64): paddle.complex64,
    (paddle.float32, paddle.complex128): paddle.complex128,
    (paddle.float64, paddle.complex64): paddle.complex128,
    (paddle.float64, paddle.complex128): paddle.complex128,
}


def _two_arg(f):
    @_wraps(f)
    def _f(x1, x2, /, **kwargs):
        x1, x2 = _fix_promotion(x1, x2)
        return f(x1, x2, **kwargs)

    if _f.__doc__ is None:
        _f.__doc__ = f"""\
Array API compatibility wrapper for paddle.{f.__name__}.

See the corresponding Paddle documentation and/or the array API specification
for more details.

"""
    return _f


def _fix_promotion(x1, x2, only_scalar=True):
    if not isinstance(x1, paddle.Tensor) or not isinstance(x2, paddle.Tensor):
        return x1, x2
    if x1.dtype not in _array_api_dtypes or x2.dtype not in _array_api_dtypes:
        return x1, x2
    # If an argument is 0-D paddle downcasts the other argument
    if not only_scalar or x1.shape == ():
        dtype = result_type(x1, x2)
        x2 = x2.to(dtype)
    if not only_scalar or x2.shape == ():
        dtype = result_type(x1, x2)
        x1 = x1.to(dtype)
    return x1, x2


def result_type(*arrays_and_dtypes: Union[array, Dtype]) -> Dtype:
    if len(arrays_and_dtypes) == 0:
        raise TypeError("At least one array or dtype must be provided")
    if len(arrays_and_dtypes) == 1:
        x = arrays_and_dtypes[0]
        if isinstance(x, paddle.dtype):
            return x
        return x.dtype
    if len(arrays_and_dtypes) > 2:
        return result_type(arrays_and_dtypes[0], result_type(*arrays_and_dtypes[1:]))

    x, y = arrays_and_dtypes
    xdt = x.dtype if not isinstance(x, paddle.dtype) else x
    ydt = y.dtype if not isinstance(y, paddle.dtype) else y

    if (xdt, ydt) in _promotion_table:
        return _promotion_table[xdt, ydt]

    # This doesn't result_type(dtype, dtype) for non-array API dtypes
    # because paddle.result_type only accepts tensors. This does however, allow
    # cross-kind promotion.
    x = paddle.to_tensor([], dtype=x) if isinstance(x, paddle.dtype) else x
    y = paddle.to_tensor([], dtype=y) if isinstance(y, paddle.dtype) else y
    return paddle.result_type(x, y)


def can_cast(from_: Union[Dtype, array], to: Dtype, /) -> bool:
    if paddle.is_tensor(from_):
        from_ = from_.dtype

    assert isinstance(from_, paddle.dtype), from_.dtype
    assert isinstance(to, paddle.dtype), to.dtype

    can_cast_dict = {
        paddle.bfloat16: {
            paddle.bfloat16: True,
            paddle.float16: True,
            paddle.float32: True,
            paddle.float64: True,
            paddle.complex64: True,
            paddle.complex128: True,
            paddle.uint8: True,
            paddle.int8: True,
            paddle.int16: True,
            paddle.int32: True,
            paddle.int64: True,
            paddle.bool: True,
        },
        paddle.float16: {
            paddle.bfloat16: True,
            paddle.float16: True,
            paddle.float32: True,
            paddle.float64: True,
            paddle.complex64: True,
            paddle.complex128: True,
            paddle.uint8: True,
            paddle.int8: True,
            paddle.int16: True,
            paddle.int32: True,
            paddle.int64: True,
            paddle.bool: True,
        },
        paddle.float32: {
            paddle.bfloat16: True,
            paddle.float16: True,
            paddle.float32: True,
            paddle.float64: True,
            paddle.complex64: True,
            paddle.complex128: True,
            paddle.uint8: True,
            paddle.int8: True,
            paddle.int16: True,
            paddle.int32: True,
            paddle.int64: True,
            paddle.bool: True,
        },
        paddle.float64: {
            paddle.bfloat16: True,
            paddle.float16: True,
            paddle.float32: True,
            paddle.float64: True,
            paddle.complex64: True,
            paddle.complex128: True,
            paddle.uint8: True,
            paddle.int8: True,
            paddle.int16: True,
            paddle.int32: True,
            paddle.int64: True,
            paddle.bool: True,
        },
        paddle.complex64: {
            paddle.bfloat16: True,
            paddle.float16: True,
            paddle.float32: True,
            paddle.float64: True,
            paddle.complex64: True,
            paddle.complex128: True,
            paddle.uint8: True,
            paddle.int8: True,
            paddle.int16: True,
            paddle.int32: True,
            paddle.int64: True,
            paddle.bool: True,
        },
        paddle.complex128: {
            paddle.bfloat16: True,
            paddle.float16: True,
            paddle.float32: True,
            paddle.float64: True,
            paddle.complex64: True,
            paddle.complex128: True,
            paddle.uint8: True,
            paddle.int8: True,
            paddle.int16: True,
            paddle.int32: True,
            paddle.int64: True,
            paddle.bool: True,
        },
        paddle.uint8: {
            paddle.bfloat16: True,
            paddle.float16: True,
            paddle.float32: True,
            paddle.float64: True,
            paddle.complex64: True,
            paddle.complex128: True,
            paddle.uint8: True,
            paddle.int8: True,
            paddle.int16: True,
            paddle.int32: True,
            paddle.int64: True,
            paddle.bool: True,
        },
        paddle.int8: {
            paddle.bfloat16: True,
            paddle.float16: True,
            paddle.float32: True,
            paddle.float64: True,
            paddle.complex64: True,
            paddle.complex128: True,
            paddle.uint8: True,
            paddle.int8: True,
            paddle.int16: True,
            paddle.int32: True,
            paddle.int64: True,
            paddle.bool: True,
        },
        paddle.int16: {
            paddle.bfloat16: True,
            paddle.float16: True,
            paddle.float32: True,
            paddle.float64: True,
            paddle.complex64: True,
            paddle.complex128: True,
            paddle.uint8: True,
            paddle.int8: True,
            paddle.int16: True,
            paddle.int32: True,
            paddle.int64: True,
            paddle.bool: True,
        },
        paddle.int32: {
            paddle.bfloat16: True,
            paddle.float16: True,
            paddle.float32: True,
            paddle.float64: True,
            paddle.complex64: True,
            paddle.complex128: True,
            paddle.uint8: True,
            paddle.int8: True,
            paddle.int16: True,
            paddle.int32: True,
            paddle.int64: True,
            paddle.bool: True,
        },
        paddle.int64: {
            paddle.bfloat16: True,
            paddle.float16: True,
            paddle.float32: True,
            paddle.float64: True,
            paddle.complex64: True,
            paddle.complex128: True,
            paddle.uint8: True,
            paddle.int8: True,
            paddle.int16: True,
            paddle.int32: True,
            paddle.int64: True,
            paddle.bool: True,
        },
        paddle.bool: {
            paddle.bfloat16: True,
            paddle.float16: True,
            paddle.float32: True,
            paddle.float64: True,
            paddle.complex64: True,
            paddle.complex128: True,
            paddle.uint8: True,
            paddle.int8: True,
            paddle.int16: True,
            paddle.int32: True,
            paddle.int64: True,
            paddle.bool: True,
        },
    }
    return can_cast_dict[from_][to]


# Basic renames
bitwise_invert = paddle.bitwise_not
newaxis = None
# paddle.conj sets the conjugation bit, which breaks conversion to other
# libraries. See https://github.com/data-apis/array-api-compat/issues/173
conj = paddle.conj

# Two-arg elementwise functions
# These require a wrapper to do the correct type promotion on 0-D tensors
add = _two_arg(paddle.add)
atan2 = _two_arg(paddle.atan2)
bitwise_and = _two_arg(paddle.bitwise_and)
bitwise_left_shift = _two_arg(paddle.bitwise_left_shift)
bitwise_or = _two_arg(paddle.bitwise_or)
bitwise_right_shift = _two_arg(paddle.bitwise_right_shift)
bitwise_xor = _two_arg(paddle.bitwise_xor)
copysign = _two_arg(paddle.copysign)
divide = _two_arg(paddle.divide)
# Also a rename. paddle.equal does not broadcast
equal = _two_arg(paddle.equal)
floor_divide = _two_arg(paddle.floor_divide)
greater = _two_arg(paddle.greater_than)
greater_equal = _two_arg(paddle.greater_equal)
hypot = _two_arg(paddle.hypot)
less = _two_arg(paddle.less)
less_equal = _two_arg(paddle.less_equal)
logaddexp = _two_arg(paddle.logaddexp)
# logical functions are not included here because they only accept bool in the
# spec, so type promotion is irrelevant.
maximum = _two_arg(paddle.maximum)
minimum = _two_arg(paddle.minimum)
multiply = _two_arg(paddle.multiply)
not_equal = _two_arg(paddle.not_equal)
pow = _two_arg(paddle.pow)
remainder = _two_arg(paddle.remainder)
subtract = _two_arg(paddle.subtract)


def max(
    x: array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> array:
    if axis == ():
        return paddle.clone(x)
    return paddle.amax(x, axis, keepdim=keepdims)


def argmax(
    x: array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> array:
    return paddle.argmax(x, axis, keepdim=keepdims)


def min(
    x: array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> array:
    if axis == ():
        return paddle.clone(x)
    return paddle.min(x, axis, keepdim=keepdims)


def argmin(
    x: array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> array:
    return paddle.argmin(x, axis, keepdim=keepdims)


unstack = get_xp(paddle)(_aliases_unstack)


# paddle.sort also returns a tuple
def sort(
    x: array,
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    **kwargs,
) -> array:
    return paddle.sort(x, axis=axis, descending=descending, stable=stable, **kwargs)


def _normalize_axes(axis, ndim):
    axes = []
    if ndim == 0 and axis:
        # Better error message in this case
        raise IndexError(f"Dimension out of range: {axis[0]}")
    lower, upper = -ndim, ndim - 1
    for a in axis:
        if a < lower or a > upper:
            # Match paddle error message (e.g., from sum())
            raise IndexError(
                f"Dimension out of range (expected to be in range of [{lower}, {upper}], but got {a}"
            )
        if a < 0:
            a = a + ndim
        if a in axes:
            # Use IndexError instead of RuntimeError, and "axis" instead of "dim"
            raise IndexError(f"Axis {a} appears multiple times in the list of axes")
        axes.append(a)
    return sorted(axes)


def _axis_none_keepdims(x, ndim, keepdims):
    # Apply keepdims when axis=None
    # Note that this is only valid for the axis=None case.
    if keepdims:
        for i in range(ndim):
            x = paddle.unsqueeze(x, 0)
    return x


def _reduce_multiple_axes(f, x, axis, keepdims=False, **kwargs):
    # Some reductions don't support multiple axes
    axes = _normalize_axes(axis, x.ndim)
    for a in reversed(axes):
        x = paddle.moveaxis(x, a, -1)
    x = paddle.flatten(x, -len(axes))

    out = f(x, -1, **kwargs)

    if keepdims:
        for a in axes:
            out = paddle.unsqueeze(out, a)
    return out


def prod(
    x: array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype: Optional[Dtype] = None,
    keepdims: bool = False,
    **kwargs,
) -> array:
    if not paddle.is_tensor(x):
        x = paddle.to_tensor(x)
    ndim = x.ndim

    # below because it still needs to upcast.
    if axis == ():
        if dtype is None:
            # We can't upcast uint8 according to the spec because there is no
            # paddle.uint64, so at least upcast to int64 which is what sum does
            # when axis=None.
            if x.dtype in [paddle.int8, paddle.int16, paddle.int32, paddle.uint8]:
                return x.to(paddle.int64)
            return x.clone()
        return x.to(dtype)

    # paddle.prod doesn't support multiple axes
    if isinstance(axis, tuple):
        return _reduce_multiple_axes(
            paddle.prod, x, axis, keepdim=keepdims, dtype=dtype, **kwargs
        )
    if axis is None:
        # paddle doesn't support keepdims with axis=None
        res = paddle.prod(x, dtype=dtype, **kwargs)
        res = _axis_none_keepdims(res, ndim, keepdims)
        return res

    return paddle.prod(x, axis, dtype=dtype, keepdim=keepdims, **kwargs)


def sum(
    x: array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    dtype: Optional[Dtype] = None,
    keepdims: bool = False,
    **kwargs,
) -> array:
    if not paddle.is_tensor(x):
        x = paddle.to_tensor(x)
    ndim = x.ndim

    # Make sure it upcasts.
    if axis == ():
        if dtype is None:
            # We can't upcast uint8 according to the spec because there is no
            # paddle.uint64, so at least upcast to int64 which is what sum does
            # when axis=None.
            if x.dtype in [paddle.int8, paddle.int16, paddle.int32, paddle.uint8]:
                return x.to(paddle.int64)
            return x.clone()
        return x.to(dtype)

    if axis is None:
        # paddle doesn't support keepdims with axis=None
        res = paddle.sum(x, dtype=dtype, **kwargs)
        res = _axis_none_keepdims(res, ndim, keepdims)
        return res

    return paddle.sum(x, axis, dtype=dtype, keepdim=keepdims, **kwargs)


def any(
    x: array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
    **kwargs,
) -> array:
    if not paddle.is_tensor(x):
        x = paddle.to_tensor(x)
    ndim = x.ndim
    if axis == ():
        return x.to(paddle.bool)
    # paddle.any doesn't support multiple axes
    if isinstance(axis, tuple):
        res = _reduce_multiple_axes(paddle.any, x, axis, keepdim=keepdims, **kwargs)
        return res.to(paddle.bool)
    if axis is None:
        # paddle doesn't support keepdims with axis=None
        res = paddle.any(x, **kwargs)
        res = _axis_none_keepdims(res, ndim, keepdims)
        return res.to(paddle.bool)

    # paddle.any doesn't return bool for uint8
    return paddle.any(x, axis, keepdim=keepdims).to(paddle.bool)


def all(
    x: array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
    **kwargs,
) -> array:
    if not paddle.is_tensor(x):
        x = paddle.to_tensor(x)
    ndim = x.ndim
    if axis == ():
        return x.to(paddle.bool)
    # paddle.all doesn't support multiple axes
    if isinstance(axis, tuple):
        res = _reduce_multiple_axes(paddle.all, x, axis, keepdim=keepdims, **kwargs)
        return res.to(paddle.bool)
    if axis is None:
        # paddle doesn't support keepdims with axis=None
        res = paddle.all(x, **kwargs)
        res = _axis_none_keepdims(res, ndim, keepdims)
        return res.to(paddle.bool)

    # paddle.all doesn't return bool for uint8
    return paddle.all(x, axis, keepdim=keepdims).to(paddle.bool)


def mean(
    x: array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
    **kwargs,
) -> array:
    if axis == ():
        return paddle.clone(x)
    if axis is None:
        # paddle doesn't support keepdims with axis=None
        res = paddle.mean(x, **kwargs)
        res = _axis_none_keepdims(res, x.ndim, keepdims)
        return res
    return paddle.mean(x, axis, keepdim=keepdims, **kwargs)


def std(
    x: array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
    **kwargs,
) -> array:
    # Note, float correction is not supported
    # implement it here for now.

    if isinstance(correction, float):
        _correction = int(correction)
        if correction != _correction:
            raise NotImplementedError(
                "float correction in paddle std() is not yet supported"
            )
    elif isinstance(correction, int):
        if correction not in [0, 1]:
            raise NotImplementedError("correction only can be 0 or 1")
    elif not isinstance(correction, bool):
        raise NotImplementedError("Only support bool correction and 0, 1")

    _correction = bool(_correction)

    if axis == ():
        return paddle.zeros_like(x)
    if isinstance(axis, int):
        axis = (axis,)
    if axis is None:
        # paddle doesn't support keepdims with axis=None
        res = paddle.std(x, tuple(range(x.ndim)), unbiased=_correction, **kwargs)
        res = _axis_none_keepdims(res, x.ndim, keepdims)
        return res
    return paddle.std(x, axis, unbiased=_correction, keepdim=keepdims, **kwargs)


def var(
    x: array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    correction: Union[int, float] = 0.0,
    keepdims: bool = False,
    **kwargs,
) -> array:
    # Note, float correction is not supported
    # implement it here for now.

    # if isinstance(correction, float):
    #     correction = int(correction)
    if isinstance(correction, float):
        _correction = int(correction)
        if correction != _correction:
            raise NotImplementedError(
                "float correction in paddle std() is not yet supported"
            )
    elif isinstance(correction, int):
        if correction not in [0, 1]:
            raise NotImplementedError("correction only can be 0 or 1")
    elif not isinstance(correction, bool):
        raise NotImplementedError("Only support bool correction and 0, 1")

    _correction = bool(_correction)

    if axis == ():
        return paddle.zeros_like(x)
    if isinstance(axis, int):
        axis = (axis,)
    if axis is None:
        # paddle doesn't support keepdims with axis=None
        res = paddle.var(x, tuple(range(x.ndim)), unbiased=_correction, **kwargs)
        res = _axis_none_keepdims(res, x.ndim, keepdims)
        return res
    return paddle.var(x, axis, unbiased=_correction, keepdim=keepdims, **kwargs)


# paddle.concat doesn't support dim=None
def concat(
    arrays: Union[Tuple[array, ...], List[array]],
    /,
    *,
    axis: Optional[int] = 0,
    **kwargs,
) -> array:
    if axis is None:
        arrays = tuple(ar.flatten() for ar in arrays)
        axis = 0
    return paddle.concat(arrays, axis, **kwargs)


def squeeze(x: array, /, axis: Union[int, Tuple[int, ...]]) -> array:
    if isinstance(axis, int):
        axis = (axis,)
    for a in axis:
        if x.shape[a] != 1:
            raise ValueError("squeezed dimensions must be equal to 1")
    axes = _normalize_axes(axis, x.ndim)

    sequence = [a - i for i, a in enumerate(axes)]
    for a in sequence:
        x = paddle.squeeze(x, a)
    return x


# paddle.broadcast_to uses size instead of shape
def broadcast_to(x: array, /, shape: Tuple[int, ...], **kwargs) -> array:
    return paddle.broadcast_to(x, shape, **kwargs)


# paddle.permute uses dims instead of axes
def permute_dims(x: array, /, axes: Tuple[int, ...]) -> array:
    return paddle.transpose(x, axes)


# The axis parameter doesn't work for flip() and roll()
# accept axis=None
def flip(
    x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, **kwargs
) -> array:
    if axis is None:
        axis = tuple(range(x.ndim))
    # paddle.flip doesn't accept dim as an int but the method does
    return x.flip(axis, **kwargs)


def roll(
    x: array,
    /,
    shift: Union[int, Tuple[int, ...]],
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    **kwargs,
) -> array:
    return paddle.roll(x, shift, axis, **kwargs)


def nonzero(x: array, /, **kwargs) -> Tuple[array, ...]:
    if x.ndim == 0:
        raise ValueError("nonzero() does not support zero-dimensional arrays")
    return paddle.nonzero(x, as_tuple=True, **kwargs)


def where(condition: array, x1: array, x2: array, /) -> array:
    x1, x2 = _fix_promotion(x1, x2)
    return paddle.where(condition, x1, x2)


def empty_like(
    x: array, /, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None
) -> array:
    out = paddle.empty_like(x, dtype=dtype)
    if device is not None:
        out = out.to(device)
    return out


def zeros_like(
    x: array, /, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None
) -> array:
    out = paddle.zeros_like(x, dtype=dtype)
    if device is not None:
        out = out.to(device)
    return out


def ones_like(
    x: array, /, *, dtype: Optional[Dtype] = None, device: Optional[Device] = None
) -> array:
    out = paddle.ones_like(x, dtype=dtype)
    if device is not None:
        out = out.to(device)
    return out


def full_like(
    x: array,
    /,
    fill_value: bool | int | float | complex,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
) -> array:
    out = paddle.full_like(x, fill_value, dtype=dtype)
    if device is not None:
        out = out.to(device)
    return out


# paddle.reshape doesn't have the copy keyword
def reshape(
    x: array, /, shape: Tuple[int, ...], copy: Optional[bool] = None, **kwargs
) -> array:
    return paddle.reshape(x, shape, **kwargs)


# paddle.arange doesn't support returning empty arrays
# keyword argument combinations
def arange(
    start: Union[int, float],
    /,
    stop: Optional[Union[int, float]] = None,
    step: Union[int, float] = 1,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    **kwargs,
) -> array:
    return paddle.arange(start, stop, step, dtype=dtype, **kwargs).to(device)


# paddle.eye does not accept None as a default for the second argument and
def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    /,
    *,
    k: int = 0,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    **kwargs,
) -> array:
    if n_cols is None:
        n_cols = n_rows
    z = paddle.zeros([n_rows, n_cols], dtype=dtype, **kwargs).to(device)
    if abs(k) <= n_rows + n_cols:
        z.diagonal(k).fill_(1)
    return z


# paddle.linspace doesn't have the endpoint parameter
def linspace(
    start: Union[int, float],
    stop: Union[int, float],
    /,
    num: int,
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    endpoint: bool = True,
    **kwargs,
) -> array:
    if not endpoint:
        return paddle.linspace(start, stop, num + 1, dtype=dtype, **kwargs).to(device)[
            :-1
        ]
    return paddle.linspace(start, stop, num, dtype=dtype, **kwargs).to(device)


# paddle.full does not accept an int size
def full(
    shape: Union[int, Tuple[int, ...]],
    fill_value: Union[bool, int, float, complex],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    **kwargs,
) -> array:
    if isinstance(shape, int):
        shape = (shape,)

    return paddle.full(shape, fill_value, dtype=dtype, **kwargs).to(device)


# ones, zeros, and empty do not accept shape as a keyword argument
def ones(
    shape: Union[int, Tuple[int, ...]],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    **kwargs,
) -> array:
    return paddle.ones(shape, dtype=dtype, **kwargs).to(device)


def zeros(
    shape: Union[int, Tuple[int, ...]],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    **kwargs,
) -> array:
    return paddle.zeros(shape, dtype=dtype, **kwargs).to(device)


def empty(
    shape: Union[int, Tuple[int, ...]],
    *,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    **kwargs,
) -> array:
    return paddle.empty(shape, dtype=dtype, **kwargs).to(device)


# tril and triu do not call the keyword argument k


def tril(x: array, /, *, k: int = 0) -> array:
    return paddle.tril(x, k)


def triu(x: array, /, *, k: int = 0) -> array:
    return paddle.triu(x, k)


def expand_dims(x: array, /, *, axis: int = 0) -> array:
    return paddle.unsqueeze(x, axis)


def astype(
    x: array, dtype: Dtype, /, *, copy: bool = True, device: Optional[Device] = None
) -> array:
    # if copy is not None:
    #     raise NotImplementedError("paddle.astype doesn't yet support the copy keyword")
    t = x.to(dtype, device=device)
    if copy:
        t = t.detach().clone()
    return t


def broadcast_arrays(*arrays: array) -> List[array]:
    return paddle.broadcast_tensors(arrays)


# Note that these named tuples aren't actually part of the standard namespace,
# but I don't see any issue with exporting the names here regardless.
from ..common._aliases import UniqueAllResult, UniqueCountsResult, UniqueInverseResult


def unique_all(x: array) -> UniqueAllResult:
    return paddle.unique(
        x,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )


def unique_counts(x: array) -> UniqueCountsResult:
    values, counts = paddle.unique(x, return_counts=True)

    # paddle.unique incorrectly gives a 0 count for nan values.
    counts[paddle.isnan(values)] = 1
    return UniqueCountsResult(values, counts)


def unique_inverse(x: array) -> UniqueInverseResult:
    values, inverse = paddle.unique(x, return_inverse=True)
    return UniqueInverseResult(values, inverse)


def unique_values(x: array) -> array:
    return paddle.unique(x)


def matmul(x1: array, x2: array, /, **kwargs) -> array:
    # paddle.matmul doesn't type promote (but differently from _fix_promotion)
    x1, x2 = _fix_promotion(x1, x2, only_scalar=False)
    return paddle.matmul(x1, x2, **kwargs)


def meshgrid(*arrays: array, indexing: str = "xy") -> List[array]:
    if indexing == "ij":
        return paddle.meshgrid(*arrays)
    else:
        return [i.T for i in paddle.meshgrid(*arrays)]


matrix_transpose = paddle.linalg.matrix_transpose


def vecdot(x1: array, x2: array, /, *, axis: int = -1) -> array:
    x1, x2 = _fix_promotion(x1, x2, only_scalar=False)
    return paddle.linalg.vecdot(x1, x2, axis=axis)


# paddle.tensordot uses dims instead of axes
def tensordot(
    x1: array,
    x2: array,
    /,
    *,
    axes: Union[int, Tuple[Sequence[int], Sequence[int]]] = 2,
    **kwargs,
) -> array:
    # Note: paddle.tensordot fails with integer dtypes when there is only 1
    x1, x2 = _fix_promotion(x1, x2, only_scalar=False)
    return paddle.tensordot(x1, x2, axes=axes, **kwargs)


def isdtype(
    dtype: Dtype,
    kind: Union[Dtype, str, Tuple[Union[Dtype, str], ...]],
    *,
    _tuple=True,  # Disallow nested tuples
) -> bool:
    """
    Returns a boolean indicating whether a provided dtype is of a specified data type ``kind``.

    Note that outside of this function, this compat library does not yet fully
    support complex numbers.

    See
    https://data-apis.org/array-api/latest/API_specification/generated/array_api.isdtype.html
    for more details
    """

    def is_signed(dtype):
        return dtype in [paddle.int8, paddle.int16, paddle.int32, paddle.int64]

    def is_complex(dtype):
        return dtype in [paddle.complex64, paddle.complex128]

    if isinstance(kind, tuple) and _tuple:
        return _builtin_any(isdtype(dtype, k, _tuple=False) for k in kind)

    elif isinstance(kind, str):
        if kind == "bool":
            return dtype == paddle.bool
        elif kind == "signed integer":
            return dtype in _int_dtypes and is_signed(dtype)
        elif kind == "unsigned integer":
            return dtype in _int_dtypes and not is_signed(dtype)
        elif kind == "integral":
            return dtype in _int_dtypes
        elif kind == "real floating":
            return dtype in [
                paddle.framework.core.VarDesc.VarType.FP32,
                paddle.framework.core.VarDesc.VarType.FP64,
                paddle.framework.core.VarDesc.VarType.FP16,
                paddle.framework.core.VarDesc.VarType.BF16,
                paddle.framework.core.DataType.FLOAT32,
                paddle.framework.core.DataType.FLOAT64,
                paddle.framework.core.DataType.FLOAT16,
                paddle.framework.core.DataType.BFLOAT16,
        ]
        elif kind == "complex floating":
            return is_complex(dtype)
        elif kind == "numeric":
            return isdtype(dtype, ("integral", "real floating", "complex floating"))
        else:
            raise ValueError(f"Unrecognized data type kind: {kind!r}")
    else:
        return dtype == kind


def take(x: array, indices: array, /, *, axis: Optional[int] = None, **kwargs) -> array:
    if axis is None:
        if x.ndim != 1:
            raise ValueError("axis must be specified when ndim > 1")
        axis = 0
    return paddle.index_select(x, axis, indices, **kwargs)


def sign(x: array, /) -> array:
    # paddle sign() does not support complex numbers and does not propagate
    # nans. See https://github.com/data-apis/array-api-compat/issues/136
    if paddle.is_complex(x):
        out = x / paddle.abs(x)
        # sign(0) = 0 but the above formula would give nan
        out[x == 0 + 0j] = 0 + 0j
        return out
    else:
        out = paddle.sign(x)
        if paddle.is_floating_point(x):
            out = paddle.where(paddle.isnan(x), paddle.full(x.shape, paddle.nan), out)
        return out


def broadcast_shapes(*shapes: List[int]) -> List[int]:
    out_shape = shapes[0]
    for i, shape in enumerate(shapes):
        if i == 0:
            continue
        out_shape = paddle.broadcast_shape(out_shape, shape)

    return out_shape


# asarray also adds the copy keyword, which is not present in numpy 1.0.
def asarray(
    obj: Union[
        array,
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
    copy: Optional[bool] = None,
    **kwargs,
) -> array:
    """
    Array API compatibility wrapper for asarray().

    See the corresponding documentation in the array library and/or the array API
    specification for more details.
    """
    if copy is False:
        if hasattr(obj, "__dlpack__"):
            obj = paddle.from_dlpack(obj.__dlpack__())
            if device is not None:
                obj = obj.to(device)
            if dtype is not None:
                obj = obj.to(dtype)
            return obj
        else:
            raise NotImplementedError(
                "asarray(obj, ..., copy=False) is not supported "
                "for obj do not has '__dlpack__()' method"
            )
    elif copy is True:
        obj = np.array(obj, copy=True)
        if np.issubdtype(obj.dtype, np.floating) and dtype is None:
            obj = obj.astype(paddle.get_default_dtype())
        return paddle.to_tensor(obj, dtype=dtype, place=device)
    else:
        if not paddle.is_tensor(obj) or (dtype is not None and obj.dtype != dtype):
            obj = np.array(obj, copy=False)
            if np.issubdtype(obj.dtype, np.floating) and dtype is None:
                obj = obj.astype(paddle.get_default_dtype())
            if dtype != paddle.bool and dtype != "bool":
                obj = paddle.from_dlpack(obj.__dlpack__(), **kwargs).to(dtype)
            else:
                obj = paddle.to_tensor(obj, dtype=dtype)
            if device is not None:
                obj = obj.to(device)
            return obj

        return obj


def floor(x: array, /) -> array:
    return paddle.floor(x).to(x.dtype)


def ceil(x: array, /) -> array:
    return paddle.ceil(x).to(x.dtype)


def clip(
    x: array,
    /,
    min: Optional[Union[int, float, array]] = None,
    max: Optional[Union[int, float, array]] = None,
) -> array:
    if min is None and max is None:
        return x

    def _isscalar(a):
        return isinstance(a, (int, float, type(None)))

    min_shape = [] if _isscalar(min) else min.shape
    max_shape = [] if _isscalar(max) else max.shape

    result_shape = broadcast_shapes(x.shape, min_shape, max_shape)

    # np.clip does type promotion but the array API clip requires that the
    # output have the same dtype as x. We do this instead of just downcasting
    # the result of xp.clip() to handle some corner cases better (e.g.,
    # avoiding uint64 -> float64 promotion).

    # Note: cases where min or max overflow (integer) or round (float) in the
    # wrong direction when downcasting to x.dtype are unspecified. This code
    # just does whatever NumPy does when it downcasts in the assignment, but
    # other behavior could be preferred, especially for integers. For example,
    # this code produces:

    # >>> clip(asarray(0, dtype=int8), asarray(128, dtype=int16), None)
    # -128

    # but an answer of 0 might be preferred. See
    # https://github.com/numpy/numpy/issues/24976 for more discussion on this issue.

    # At least handle the case of Python integers correctly (see
    # https://github.com/numpy/numpy/pull/26892).
    if type(min) is int and min <= paddle.iinfo(x.dtype).min:
        min = None
    if type(max) is int and max >= paddle.iinfo(x.dtype).max:
        max = None

    out = paddle.to_tensor(broadcast_to(x, result_shape), place=x.place)
    if min is not None:
        if paddle.is_tensor(x) and x.dtype == paddle.float64 and _isscalar(min):
            # Avoid loss of precision due to paddle defaulting to float32
            min = paddle.to_tensor(min, dtype=paddle.float64)
        a = broadcast_to(paddle.to_tensor(min, place=x.place), result_shape)
        ia = (out < a) | paddle.isnan(a)
        # paddle requires an explicit cast here
        out[ia] = astype(a[ia], out.dtype)
    if max is not None:
        if paddle.is_tensor(x) and x.dtype == paddle.float64 and _isscalar(max):
            max = paddle.to_tensor(max, dtype=paddle.float64)
        b = broadcast_to(paddle.to_tensor(max, place=x.place), result_shape)
        ib = (out > b) | paddle.isnan(b)
        out[ib] = astype(b[ib], out.dtype)
    # Return a scalar for 0-D
    return out


def cumulative_sum(
    x: array,
    /,
    *,
    axis: Optional[int] = None,
    dtype: Optional[Dtype] = None,
    include_initial: bool = False,
) -> array:
    if axis is None:
        if x.ndim > 1:
            raise ValueError(
                "axis must be specified in cumulative_sum for more than one dimension"
            )
        axis = 0

    res = paddle.cumsum(x, axis=axis, dtype=dtype)

    # np.cumsum does not support include_initial
    if include_initial:
        initial_shape = list(x.shape)
        initial_shape[axis] = 1
        res = paddle.concat(
            [paddle.zeros(shape=initial_shape, dtype=res.dtype).to(res.place), res],
            axis=axis,
        )
    return res


def searchsorted(
    x1: array,
    x2: array,
    /,
    *,
    side: Literal["left", "right"] = "left",
    sorter: array | None = None,
) -> array:
    if sorter is None:
        return paddle.searchsorted(x1, x2, right=(side == "right"))

    return paddle.searchsorted(
        x1.take_along_axis(axis=-1, indices=sorter),
        x2,
        right=(side == "right"),
    )


__all__ = [
    "__array_namespace_info__",
    "result_type",
    "can_cast",
    "permute_dims",
    "bitwise_invert",
    "newaxis",
    "conj",
    "add",
    "atan2",
    "bitwise_and",
    "bitwise_left_shift",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "copysign",
    "divide",
    "equal",
    "floor_divide",
    "greater",
    "greater_equal",
    "hypot",
    "less",
    "less_equal",
    "logaddexp",
    "maximum",
    "minimum",
    "multiply",
    "not_equal",
    "pow",
    "remainder",
    "subtract",
    "max",
    "min",
    "clip",
    "unstack",
    "cumulative_sum",
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
    "UniqueAllResult",
    "UniqueCountsResult",
    "UniqueInverseResult",
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
    "sign",
    "broadcast_shapes",
    "argmax",
    "argmin",
    "searchsorted",
    "empty_like",
    "zeros_like",
    "ones_like",
    "full_like",
    "asarray",
    "ceil",
    "floor",
]

_all_ignore = ["paddle", "get_xp"]
