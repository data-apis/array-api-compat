from __future__ import annotations

import tensorflow as tf

from .._internal import clone_module
from ..common._linalg import (
    EigResult,
    EighResult,
    QRResult,
    SlogdetResult,
    SVDResult,
)
from ..common._typing import JustFloat, JustInt
from ._aliases import (
    _is_complex,
    _moveaxis,
    _promote_two,
    _real_dtype_for,
    _shape_tuple,
    abs,
    count_nonzero,
    finfo,
    matrix_transpose,
    sum,
)
from ._typing import Array, DType

__all__ = clone_module("tensorflow.linalg", globals())

# These functions are in both the main and linalg namespaces.
from ._aliases import matmul, tensordot, vecdot  # noqa: F401


def _replace_nonfinite(x: Array) -> Array:
    if _is_complex(x.dtype):
        finite = tf.math.is_finite(tf.math.real(x)) & tf.math.is_finite(tf.math.imag(x))
    else:
        finite = tf.math.is_finite(x)
    return tf.where(finite, x, tf.zeros((), dtype=x.dtype))


def outer(x1: Array, x2: Array, /) -> Array:
    x1, x2 = _promote_two(x1, x2)
    return tf.reshape(x1, (-1, 1)) * tf.reshape(x2, (1, -1))


def cross(x1: Array, x2: Array, /, *, axis: int = -1) -> Array:
    x1, x2 = _promote_two(x1, x2)
    if x1.shape[axis] != 3 or x2.shape[axis] != 3:
        raise ValueError("cross product axis must have size 3")
    x1 = _moveaxis(x1, axis, -1)
    x2 = _moveaxis(x2, axis, -1)
    shape = tf.broadcast_static_shape(x1.shape, x2.shape)
    x1, x2 = tf.broadcast_to(x1, shape), tf.broadcast_to(x2, shape)
    return _moveaxis(tf.linalg.cross(x1, x2), -1, axis)


def eigh(x: Array, /) -> EighResult:
    return EighResult(*tf.linalg.eigh(x))


def eig(x: Array, /) -> EigResult:
    return EigResult(*tf.linalg.eig(x))


def qr(x: Array, /, *, mode: str = "reduced") -> QRResult:
    if mode not in ("reduced", "complete"):
        raise ValueError("mode must be 'reduced' or 'complete'")
    x = _replace_nonfinite(x)
    res = tf.linalg.qr(x, full_matrices=mode == "complete")
    return QRResult(_replace_nonfinite(res.q), _replace_nonfinite(res.r))


def slogdet(x: Array, /) -> SlogdetResult:
    res = tf.linalg.slogdet(x)
    return SlogdetResult(res.sign, tf.math.real(res.log_abs_determinant))


def svd(x: Array, /, *, full_matrices: bool = True) -> SVDResult:
    s, u, v = tf.linalg.svd(x, full_matrices=full_matrices, compute_uv=True)
    vh = matrix_transpose(tf.math.conj(v))
    return SVDResult(u, s, vh)


def cholesky(x: Array, /, *, upper: bool = False) -> Array:
    out = tf.linalg.cholesky(x)
    if upper:
        out = matrix_transpose(out)
        if _is_complex(out.dtype):
            out = tf.math.conj(out)
    return out


def matrix_rank(x: Array, /, *, rtol: float | Array | None = None) -> Array:
    if x.ndim < 2:
        raise ValueError("1-dimensional array given. Array must be at least two-dimensional")
    s = svdvals(x)
    if rtol is None:
        tol = tf.reduce_max(s, axis=-1, keepdims=True) * max(x.shape[-2:]) * finfo(s.dtype).eps
    else:
        tol = tf.reduce_max(s, axis=-1, keepdims=True) * tf.cast(rtol, s.dtype)[..., tf.newaxis]
    return count_nonzero(s > tol, axis=-1)


def pinv(x: Array, /, *, rtol: float | Array | None = None) -> Array:
    s, u, v = tf.linalg.svd(x, full_matrices=False, compute_uv=True)
    if rtol is None:
        rtol = max(x.shape[-2:]) * finfo(x.dtype).eps
    rtol = tf.cast(rtol, s.dtype)
    if rtol.ndim != 0:
        rtol = rtol[..., tf.newaxis]
    cutoff = tf.reduce_max(s, axis=-1, keepdims=True) * rtol
    s_inv = tf.where(s > cutoff, tf.math.reciprocal(s), tf.zeros((), dtype=s.dtype))
    v_scaled = v * tf.cast(s_inv[..., tf.newaxis, :], v.dtype)
    return matmul(v_scaled, matrix_transpose(tf.math.conj(u)))


def matrix_norm(
    x: Array,
    /,
    *,
    keepdims: bool = False,
    ord: int | float | str | None = "fro",
) -> Array:
    out_dtype = _real_dtype_for(x.dtype)
    abs_x = tf.cast(tf.abs(x), out_dtype)

    if ord in (None, "fro"):
        out = tf.sqrt(tf.reduce_sum(tf.square(abs_x), axis=(-2, -1)))
    elif ord == 1:
        out = tf.reduce_max(tf.reduce_sum(abs_x, axis=-2), axis=-1)
    elif ord == -1:
        out = tf.reduce_min(tf.reduce_sum(abs_x, axis=-2), axis=-1)
    elif ord == float("inf"):
        out = tf.reduce_max(tf.reduce_sum(abs_x, axis=-1), axis=-1)
    elif ord == -float("inf"):
        out = tf.reduce_min(tf.reduce_sum(abs_x, axis=-1), axis=-1)
    elif ord in (2, -2, "nuc"):
        s = tf.linalg.svd(x, compute_uv=False)
        if ord == 2:
            out = tf.reduce_max(s, axis=-1)
        elif ord == -2:
            out = tf.reduce_min(s, axis=-1)
        else:
            out = tf.reduce_sum(s, axis=-1)
    else:
        raise ValueError(f"unsupported matrix norm order: {ord!r}")

    out = tf.cast(out, out_dtype)
    if keepdims:
        out = tf.reshape(out, _shape_tuple(x)[:-2] + (1, 1))
    return out


def matrix_power(x: Array, n: int, /) -> Array:
    if n == 0:
        eye = tf.eye(x.shape[-1], dtype=x.dtype)
        return tf.broadcast_to(eye, _shape_tuple(x))
    if n < 0:
        x = tf.linalg.inv(x)
        n = -n
    result = x
    for _ in range(n - 1):
        result = tf.linalg.matmul(result, x)
    return result


def solve(x1: Array, x2: Array, /) -> Array:
    x1, x2 = _promote_two(x1, x2)
    squeeze = False
    if x2.ndim == 1:
        stack_shape = _shape_tuple(x1)[:-2]
        x2 = tf.reshape(x2, (1,) * len(stack_shape) + _shape_tuple(x2) + (1,))
        x2 = tf.broadcast_to(x2, stack_shape + (_shape_tuple(x1)[-1], 1))
        squeeze = True
    else:
        stack_shape = tuple(
            tf.broadcast_static_shape(
                tf.TensorShape(_shape_tuple(x1)[:-2]),
                tf.TensorShape(_shape_tuple(x2)[:-2]),
            ).as_list()
        )
        x1 = tf.broadcast_to(x1, stack_shape + _shape_tuple(x1)[-2:])
        x2 = tf.broadcast_to(x2, stack_shape + _shape_tuple(x2)[-2:])
    out = tf.linalg.solve(x1, x2)
    return tf.squeeze(out, axis=-1) if squeeze else out


def svdvals(x: Array, /) -> Array:
    return tf.linalg.svd(x, compute_uv=False)


def diagonal(x: Array, /, *, offset: int = 0) -> Array:
    if x.ndim < 2:
        raise ValueError("x must be at least 2-dimensional for diagonal")
    if offset >= x.shape[-1] or offset <= -x.shape[-2]:
        return tf.zeros(_shape_tuple(x)[:-2] + (0,), dtype=x.dtype)
    return tf.linalg.diag_part(x, k=offset)


def trace(x: Array, /, *, offset: int = 0, dtype: DType | None = None) -> Array:
    return sum(diagonal(x, offset=offset), axis=-1, dtype=dtype)


def vector_norm(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    ord: JustInt | JustFloat = 2,
) -> Array:
    out_dtype = _real_dtype_for(x.dtype)
    if axis == ():
        return tf.cast(x != 0, out_dtype) if ord == 0 else tf.cast(abs(x), out_dtype)

    if axis is None:
        x_ = tf.reshape(x, (-1,))
        axis_ = 0
    elif isinstance(axis, tuple):
        axes = tuple(a + x.ndim if a < 0 else a for a in axis)
        rest = tuple(i for i in range(x.ndim) if i not in axes)
        x_ = tf.transpose(x, axes + rest)
        axis_size = 1
        for a in axes:
            axis_size *= x.shape[a]
        x_ = tf.reshape(x_, (axis_size, *[x.shape[i] for i in rest]))
        axis_ = 0
    else:
        x_ = x
        axis_ = axis

    abs_x = tf.abs(x_)
    abs_x = tf.cast(abs_x, out_dtype)
    if ord == 0:
        out = tf.cast(count_nonzero(x_, axis=axis_), out_dtype)
    elif ord == 1:
        out = tf.reduce_sum(abs_x, axis=axis_)
    elif ord == 2:
        out = tf.sqrt(tf.reduce_sum(tf.square(abs_x), axis=axis_))
    elif ord == float("inf"):
        out = tf.reduce_max(abs_x, axis=axis_)
    elif ord == -float("inf"):
        out = tf.reduce_min(abs_x, axis=axis_)
    else:
        p = tf.cast(ord, out_dtype)
        out = tf.pow(tf.reduce_sum(tf.pow(abs_x, p), axis=axis_), 1 / p)

    if keepdims:
        shape = list(_shape_tuple(x))
        axes = range(x.ndim) if axis is None else (axis if isinstance(axis, tuple) else (axis,))
        for a in axes:
            shape[a] = 1
        out = tf.reshape(out, shape)
    return out


__all__ = sorted(
    set(__all__)
    | {
        "EigResult",
        "EighResult",
        "QRResult",
        "SVDResult",
        "SlogdetResult",
        "cholesky",
        "cross",
        "diagonal",
        "eig",
        "eigh",
        "matmul",
        "matrix_norm",
        "matrix_power",
        "matrix_rank",
        "matrix_transpose",
        "outer",
        "pinv",
        "qr",
        "slogdet",
        "solve",
        "svd",
        "svdvals",
        "tensordot",
        "trace",
        "vecdot",
        "vector_norm",
    }
)


def __dir__() -> list[str]:
    return __all__
