# pyright: reportPrivateUsage=false
from __future__ import annotations

from builtins import bool as py_bool
from typing import Any

import mlx.core as mx

from .._internal import get_xp
from ..common import _aliases
from ..common._helpers import _check_device
from ..common._typing import NestedSequence, SupportsBufferProtocol
from ._typing import Array, Device, DType

bool = mx.bool_

# --- Renamed trig functions (MLX uses arc* like NumPy) ---
acos = mx.arccos
acosh = mx.arccosh
asin = mx.arcsin
asinh = mx.arcsinh
atan = mx.arctan
atan2 = mx.arctan2
atanh = mx.arctanh

# --- Bitwise renames (same as NumPy) ---
bitwise_left_shift = mx.left_shift
bitwise_right_shift = mx.right_shift
bitwise_invert = mx.bitwise_invert

# --- concat -> concatenate ---
concat = mx.concatenate

# --- pow ---
pow = mx.power

# --- Creation functions: delegate to common (adds device kwarg handling) ---
# arange/eye/linspace need MLX-specific overrides due to kwarg name differences.
empty = get_xp(mx)(_aliases.empty)
empty_like = get_xp(mx)(_aliases.empty_like)
full = get_xp(mx)(_aliases.full)
full_like = get_xp(mx)(_aliases.full_like)
ones = get_xp(mx)(_aliases.ones)
ones_like = get_xp(mx)(_aliases.ones_like)
zeros = get_xp(mx)(_aliases.zeros)
zeros_like = get_xp(mx)(_aliases.zeros_like)


def arange(
    start: float,
    /,
    stop: float | None = None,
    step: float = 1,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    **kwargs: Any,
) -> Array:
    """Array API wrapper: MLX arange doesn't accept stop=None as keyword."""
    _check_device(mx, device)
    if stop is None:
        # MLX arange(stop) form, step ignored when only stop given
        return mx.arange(start, dtype=dtype)
    return mx.arange(start, stop, step, dtype=dtype)


def eye(
    n_rows: int,
    n_cols: int | None = None,
    /,
    *,
    k: int = 0,
    dtype: DType | None = None,
    device: Device | None = None,
    **kwargs: Any,
) -> Array:
    """Array API wrapper: MLX eye uses 'm' not 'M'."""
    _check_device(mx, device)
    return mx.eye(n_rows, m=n_cols, k=k, dtype=dtype)


def linspace(
    start: float,
    stop: float,
    /,
    num: int = 50,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    endpoint: py_bool = True,
    **kwargs: Any,
) -> Array:
    """Array API wrapper: MLX linspace has no endpoint kwarg."""
    _check_device(mx, device)
    if not endpoint:
        # Emulate endpoint=False: generate num+1 points, drop the last
        return mx.linspace(start, stop, num + 1, dtype=dtype)[:-1]
    return mx.linspace(start, stop, num, dtype=dtype)

# --- Unique (split from mx.unique like NumPy) ---
UniqueAllResult = get_xp(mx)(_aliases.UniqueAllResult)
UniqueCountsResult = get_xp(mx)(_aliases.UniqueCountsResult)
UniqueInverseResult = get_xp(mx)(_aliases.UniqueInverseResult)
unique_all = get_xp(mx)(_aliases.unique_all)
unique_counts = get_xp(mx)(_aliases.unique_counts)
unique_inverse = get_xp(mx)(_aliases.unique_inverse)
unique_values = get_xp(mx)(_aliases.unique_values)

# --- std/var: MLX requires ddof as int, not float ---
def std(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    correction: float = 0.0,
    keepdims: py_bool = False,
    **kwargs: Any,
) -> Array:
    """Array API wrapper: MLX std requires ddof as int."""
    return mx.std(x, axis=axis, ddof=int(correction), keepdims=keepdims)


def var(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    correction: float = 0.0,
    keepdims: py_bool = False,
    **kwargs: Any,
) -> Array:
    """Array API wrapper: MLX var requires ddof as int."""
    return mx.var(x, axis=axis, ddof=int(correction), keepdims=keepdims)

# --- cumulative_sum/prod: rename from cumsum/cumprod + include_initial ---
cumulative_sum = get_xp(mx)(_aliases.cumulative_sum)
cumulative_prod = get_xp(mx)(_aliases.cumulative_prod)

def clip(
    x: Array,
    /,
    min: float | Array | None = None,
    max: float | Array | None = None,
    **kwargs: Any,
) -> Array:
    """
    Array API compatibility wrapper for clip().
    """
    if min is None and max is None:
        return mx.array(x)
    if min is None:
        return mx.minimum(x, mx.array(max, dtype=x.dtype))
    if max is None:
        return mx.maximum(x, mx.array(min, dtype=x.dtype))
    return mx.clip(x, mx.array(min, dtype=x.dtype), mx.array(max, dtype=x.dtype))


# --- permute_dims: MLX has this natively (unlike NumPy which has transpose) ---
permute_dims = mx.permute_dims


# --- reshape: MLX uses 'shape' not 'newshape', but copy kwarg unsupported ---
def reshape(
    x: Array,
    /,
    shape: tuple[int, ...],
    *,
    copy: py_bool | None = None,
) -> Array:
    """
    Array API compatibility wrapper for reshape().

    See the corresponding documentation in MLX and/or the array API
    specification for more details.
    """
    if copy is True:
        x = mx.array(x)
    return mx.reshape(x, shape)


# --- argsort/sort: add descending + stable kwargs MLX doesn't have ---
def argsort(
    x: Array,
    /,
    *,
    axis: int = -1,
    descending: py_bool = False,
    stable: py_bool = True,
) -> Array:
    """
    Array API compatibility wrapper for argsort().

    See the corresponding documentation in MLX and/or the array API
    specification for more details.
    """
    # MLX argsort has no stable/descending; stable sort is the default.
    # descending is emulated via flip.
    if not descending:
        return mx.argsort(x, axis=axis)
    res = mx.flip(
        mx.argsort(mx.flip(x, axis=axis), axis=axis),
        axis=axis,
    )
    normalised_axis = axis if axis >= 0 else x.ndim + axis
    max_i = x.shape[normalised_axis] - 1
    return max_i - res


def sort(
    x: Array,
    /,
    *,
    axis: int = -1,
    descending: py_bool = False,
    stable: py_bool = True,
) -> Array:
    """
    Array API compatibility wrapper for sort().

    See the corresponding documentation in MLX and/or the array API
    specification for more details.
    """
    # MLX sort has no stable/descending kwargs
    res = mx.sort(x, axis=axis)
    if descending:
        res = mx.flip(res, axis=axis)
    return res


# --- nonzero: must error on 0-d ---
nonzero = get_xp(mx)(_aliases.nonzero)

# --- linalg helpers ---
matmul = get_xp(mx)(_aliases.matmul)
matrix_transpose = get_xp(mx)(_aliases.matrix_transpose)
tensordot = get_xp(mx)(_aliases.tensordot)
sign = get_xp(mx)(_aliases.sign)

# --- vecdot: MLX has no native vecdot ---
vecdot = get_xp(mx)(_aliases.vecdot)

# --- isdtype: MLX has DtypeCategory but no isdtype function ---
isdtype = get_xp(mx)(_aliases.isdtype)

# --- unstack: MLX has no native unstack ---
unstack = get_xp(mx)(_aliases.unstack)

# --- finfo ---
finfo = get_xp(mx)(_aliases.finfo)


class iinfo_result:
    def __init__(self, min: int, max: int, bits: int, dtype: DType):
        self.min = min
        self.max = max
        self.bits = bits
        self.dtype = dtype

    def __repr__(self) -> str:
        return f"iinfo(min={self.min}, max={self.max}, dtype={self.dtype})"


def iinfo(type_: DType | Array, /) -> Any:
    """
    Array API compatibility wrapper for iinfo().

    MLX does not have a native iinfo(); we provide a minimal shim that returns
    an object with .min, .max, .bits, and .dtype attributes for integer dtypes.

    See the corresponding documentation in the array API specification for more
    details.
    """
    # Resolve dtype from array if needed
    if isinstance(type_, mx.array):
        dtype = type_.dtype
    else:
        dtype = type_

    # MLX has no iinfo; hand-coded table for integer dtypes it supports
    _info = {
        mx.int8:   (-(2**7),  2**7  - 1, 8,  mx.int8),
        mx.int16:  (-(2**15), 2**15 - 1, 16, mx.int16),
        mx.int32:  (-(2**31), 2**31 - 1, 32, mx.int32),
        mx.int64:  (-(2**63), 2**63 - 1, 64, mx.int64),
        mx.uint8:  (0,        2**8  - 1, 8,  mx.uint8),
        mx.uint16: (0,        2**16 - 1, 16, mx.uint16),
        mx.uint32: (0,        2**32 - 1, 32, mx.uint32),
        mx.uint64: (0,        2**64 - 1, 64, mx.uint64),
    }
    if dtype not in _info:
        raise TypeError(f"iinfo is not supported for dtype {dtype}")

    mn, mx_, bits, dt = _info[dtype]
    return iinfo_result(mn, mx_, bits, dt)


# --- asarray: MLX has no asarray; wrap mx.array with spec-compatible signature ---
def asarray(
    obj: Array | complex | NestedSequence[complex] | SupportsBufferProtocol,
    /,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    copy: py_bool | None = None,
    **kwargs: Any,
) -> Array:
    """
    Array API compatibility wrapper for asarray().
    """
    _check_device(mx, device)
    if isinstance(obj, mx.array):
        if dtype is None:
            dtype = obj.dtype
        if copy is False:
            if dtype != obj.dtype:
                raise ValueError("Cannot perform copy=False with a different dtype")
            return obj
        elif copy is None or copy is True:
            if copy is True:
                return mx.array(obj, dtype=dtype)
            else:
                if dtype == obj.dtype:
                    return obj
                return mx.array(obj, dtype=dtype)
    return mx.array(obj, dtype=dtype)


# --- astype: MLX astype has no copy kwarg ---
def astype(
    x: Array,
    dtype: DType,
    /,
    *,
    copy: py_bool = True,
    device: Device | None = None,
) -> Array:
    """
    Array API compatibility wrapper for astype().

    See the corresponding documentation in MLX and/or the array API
    specification for more details.
    """
    # MLX astype has no copy param; always returns new array
    return x.astype(dtype)


# --- take_along_axis: axis defaults to -1 in spec, required in MLX ---
def take_along_axis(x: Array, indices: Array, /, *, axis: int = -1) -> Array:
    """
    Array API compatibility wrapper for take_along_axis().

    See the corresponding documentation in MLX and/or the array API
    specification for more details.
    """
    return mx.take_along_axis(x, indices, axis=axis)


__all__ = _aliases.__all__ + [
    "asarray",
    "astype",
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bitwise_left_shift",
    "bitwise_invert",
    "bitwise_right_shift",
    "bool",
    "concat",
    "pow",
    "iinfo",
    "reshape",
    "argsort",
    "sort",
    "take_along_axis",
]


def __dir__() -> list[str]:
    return __all__
