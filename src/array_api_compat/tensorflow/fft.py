from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Literal

import tensorflow as tf

from .._internal import clone_module
from ._aliases import _moveaxis
from ._typing import Array, Device, DType

__all__ = clone_module("tensorflow.signal", globals())

_Norm = Literal["backward", "ortho", "forward"]


def _complex_dtype(dtype: DType) -> DType:
    if dtype in (tf.float64, tf.complex128):
        return tf.complex128
    return tf.complex64


def _real_dtype(dtype: DType) -> DType:
    if dtype == tf.complex128:
        return tf.float64
    return tf.float32


def _as_complex(x: Array) -> Array:
    if x.dtype.is_complex:
        return x
    return tf.cast(x, _complex_dtype(x.dtype))


def _shape_tuple(x: Array) -> tuple[int, ...]:
    return tuple(x.shape.as_list())


def _normalize_axis(axis: int, ndim: int) -> int:
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise IndexError(f"axis {axis} is out of bounds for array of dimension {ndim}")
    return axis


def _normalize_axes(axes: Sequence[int] | None, ndim: int, s: Sequence[int] | None) -> tuple[int, ...]:
    if axes is None:
        if s is None:
            axes = tuple(range(ndim))
        else:
            axes = tuple(range(ndim - len(s), ndim))
    axes = tuple(_normalize_axis(a, ndim) for a in axes)
    if len(set(axes)) != len(axes):
        raise ValueError("repeated axis")
    return axes


def _resize_axis(x: Array, n: int | None, axis: int) -> Array:
    if n is None or x.shape[axis] == n:
        return x
    shape = list(_shape_tuple(x))
    current = shape[axis]
    if current > n:
        begin = [0] * x.ndim
        size = shape
        size[axis] = n
        return tf.slice(x, begin, size)
    paddings = [[0, 0] for _ in range(x.ndim)]
    paddings[axis][1] = n - current
    return tf.pad(x, paddings)


def _apply_1d(x: Array, func, n: int | None, axis: int) -> Array:
    x = _moveaxis(x, axis, -1)
    x = _resize_axis(x, n, -1)
    x = func(x)
    return _moveaxis(x, -1, axis)


def _norm_size(x: Array, axes: tuple[int, ...], s: Sequence[int] | None) -> int:
    if s is None:
        return math.prod(x.shape[a] for a in axes)
    return math.prod(s)


def _scale_forward(x: Array, n: int, norm: _Norm) -> Array:
    if norm == "backward":
        return x
    scale = tf.cast(n if norm == "forward" else math.sqrt(n), x.dtype)
    return x / scale


def _scale_inverse(x: Array, n: int, norm: _Norm) -> Array:
    if norm == "backward":
        return x
    scale = tf.cast(n if norm == "forward" else math.sqrt(n), x.dtype)
    return x * scale


def fft(
    x: Array,
    /,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm = "backward",
) -> Array:
    x = _as_complex(x)
    axis = _normalize_axis(axis, x.ndim)
    out = _apply_1d(x, tf.signal.fft, n, axis)
    return _scale_forward(out, out.shape[axis], norm)


def ifft(
    x: Array,
    /,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm = "backward",
) -> Array:
    x = _as_complex(x)
    axis = _normalize_axis(axis, x.ndim)
    out = _apply_1d(x, tf.signal.ifft, n, axis)
    return _scale_inverse(out, out.shape[axis], norm)


def fftn(
    x: Array,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _Norm = "backward",
) -> Array:
    x = _as_complex(x)
    axes_ = _normalize_axes(axes, x.ndim, s)
    sizes = [None] * len(axes_) if s is None else list(s)
    n = _norm_size(x, axes_, s)
    out = x
    for axis, size in zip(axes_, sizes, strict=True):
        out = _apply_1d(out, tf.signal.fft, size, axis)
    return _scale_forward(out, n, norm)


def ifftn(
    x: Array,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _Norm = "backward",
) -> Array:
    x = _as_complex(x)
    axes_ = _normalize_axes(axes, x.ndim, s)
    sizes = [None] * len(axes_) if s is None else list(s)
    n = _norm_size(x, axes_, s)
    out = x
    for axis, size in zip(axes_, sizes, strict=True):
        out = _apply_1d(out, tf.signal.ifft, size, axis)
    return _scale_inverse(out, n, norm)


def rfft(
    x: Array,
    /,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm = "backward",
) -> Array:
    axis = _normalize_axis(axis, x.ndim)
    out = _apply_1d(x, tf.signal.rfft, n, axis)
    size = n if n is not None else x.shape[axis]
    return _scale_forward(out, size, norm)


def irfft(
    x: Array,
    /,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm = "backward",
) -> Array:
    axis = _normalize_axis(axis, x.ndim)
    out = _apply_1d(x, lambda y: tf.signal.irfft(y, fft_length=[n] if n is not None else None), None, axis)
    size = n if n is not None else out.shape[axis]
    return _scale_inverse(out, size, norm)


def rfftn(
    x: Array,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _Norm = "backward",
) -> Array:
    axes_ = _normalize_axes(axes, x.ndim, s)
    sizes = [None] * len(axes_) if s is None else list(s)
    n = _norm_size(x, axes_, s)
    out = _apply_1d(x, tf.signal.rfft, sizes[-1], axes_[-1])
    for axis, size in zip(axes_[:-1], sizes[:-1], strict=True):
        out = _apply_1d(_as_complex(out), tf.signal.fft, size, axis)
    return _scale_forward(out, n, norm)


def irfftn(
    x: Array,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _Norm = "backward",
) -> Array:
    axes_ = _normalize_axes(axes, x.ndim, s)
    sizes = [None] * len(axes_) if s is None else list(s)
    if s is None:
        last_size = 2 * (x.shape[axes_[-1]] - 1)
        n = math.prod([*(x.shape[a] for a in axes_[:-1]), last_size])
    else:
        n = math.prod(s)
    out = x
    for axis, size in zip(axes_[:-1], sizes[:-1], strict=True):
        out = _apply_1d(out, tf.signal.ifft, size, axis)
    out = _apply_1d(out, lambda y: tf.signal.irfft(y, fft_length=[sizes[-1]] if sizes[-1] is not None else None), None, axes_[-1])
    return _scale_inverse(out, n, norm)


def hfft(
    x: Array,
    /,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm = "backward",
) -> Array:
    size = n if n is not None else 2 * (x.shape[axis] - 1)
    return irfft(tf.math.conj(x), n=size, axis=axis, norm=norm) * tf.cast(size, _real_dtype(x.dtype))


def ihfft(
    x: Array,
    /,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm = "backward",
) -> Array:
    size = n if n is not None else x.shape[axis]
    return tf.math.conj(rfft(x, n=size, axis=axis, norm=norm)) / tf.cast(size, _complex_dtype(x.dtype))


def fftfreq(
    n: int,
    /,
    *,
    d: float = 1.0,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Array:
    with tf.device(device):
        dtype = dtype or tf.float32
        positive = tf.range(0, (n - 1) // 2 + 1, dtype=dtype)
        negative = tf.range(-(n // 2), 0, dtype=dtype)
        return tf.concat([positive, negative], axis=0) / tf.cast(n * d, dtype)


def rfftfreq(
    n: int,
    /,
    *,
    d: float = 1.0,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Array:
    with tf.device(device):
        dtype = dtype or tf.float32
        return tf.range(0, n // 2 + 1, dtype=dtype) / tf.cast(n * d, dtype)


def fftshift(
    x: Array,
    /,
    *,
    axes: int | Sequence[int] | None = None,
) -> Array:
    axes_ = _normalize_axes(None if axes is None else (axes if isinstance(axes, Sequence) else (axes,)), x.ndim, None)
    shifts = tuple(x.shape[axis] // 2 for axis in axes_)
    return tf.roll(x, shifts, axes_)


def ifftshift(
    x: Array,
    /,
    *,
    axes: int | Sequence[int] | None = None,
) -> Array:
    axes_ = _normalize_axes(None if axes is None else (axes if isinstance(axes, Sequence) else (axes,)), x.ndim, None)
    shifts = tuple(-(x.shape[axis] // 2) for axis in axes_)
    return tf.roll(x, shifts, axes_)


__all__ = sorted(
    set(__all__)
    | {
        "fft",
        "ifft",
        "fftn",
        "ifftn",
        "rfft",
        "irfft",
        "rfftn",
        "irfftn",
        "hfft",
        "ihfft",
        "fftfreq",
        "rfftfreq",
        "fftshift",
        "ifftshift",
    }
)


def __dir__() -> list[str]:
    return __all__
