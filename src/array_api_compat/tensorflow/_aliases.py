from __future__ import annotations

from builtins import all as py_all
from builtins import any as py_any
from builtins import abs as py_abs
from builtins import bool as py_bool
from builtins import max as py_max
from collections.abc import Sequence
from functools import reduce as _reduce, wraps as _wraps
import math
from typing import Any, Literal

import tensorflow as tf

from ..common._typing import NestedSequence, SupportsBufferProtocol
from ._typing import Array, Device, DType

bool = tf.bool
newaxis = None
e = math.e
inf = math.inf
nan = float("nan")
pi = math.pi

_py_scalars = (py_bool, int, float, complex)

_bool_dtypes = {tf.bool}
_signed_dtypes = {tf.int8, tf.int16, tf.int32, tf.int64}
_unsigned_dtypes = {tf.uint8, tf.uint16, tf.uint32, tf.uint64}
_real_floating_dtypes = {tf.float16, tf.bfloat16, tf.float32, tf.float64}
_complex_floating_dtypes = {tf.complex64, tf.complex128}
_integral_dtypes = _signed_dtypes | _unsigned_dtypes
_numeric_dtypes = _integral_dtypes | _real_floating_dtypes | _complex_floating_dtypes
_all_dtypes = _bool_dtypes | _numeric_dtypes

_dtype_bits = {
    tf.bool: 1,
    tf.int8: 8,
    tf.int16: 16,
    tf.int32: 32,
    tf.int64: 64,
    tf.uint8: 8,
    tf.uint16: 16,
    tf.uint32: 32,
    tf.uint64: 64,
    tf.float16: 16,
    tf.bfloat16: 16,
    tf.float32: 32,
    tf.float64: 64,
    tf.complex64: 64,
    tf.complex128: 128,
}

_float_for_bits = {
    16: tf.float16,
    32: tf.float32,
    64: tf.float64,
}

_complex_for_bits = {
    32: tf.complex64,
    64: tf.complex128,
}

_finfo = {
    tf.float16: {
        "bits": 16,
        "eps": 0.0009765625,
        "max": 65504.0,
        "min": -65504.0,
        "smallest_normal": 0.00006103515625,
    },
    tf.bfloat16: {
        "bits": 16,
        "eps": 0.0078125,
        "max": 3.3895313892515355e38,
        "min": -3.3895313892515355e38,
        "smallest_normal": 1.1754943508222875e-38,
    },
    tf.float32: {
        "bits": 32,
        "eps": 1.1920928955078125e-07,
        "max": 3.4028234663852886e38,
        "min": -3.4028234663852886e38,
        "smallest_normal": 1.1754943508222875e-38,
    },
    tf.float64: {
        "bits": 64,
        "eps": 2.220446049250313e-16,
        "max": 1.7976931348623157e308,
        "min": -1.7976931348623157e308,
        "smallest_normal": 2.2250738585072014e-308,
    },
}

_iinfo = {
    tf.int8: {"bits": 8, "min": -128, "max": 127},
    tf.int16: {"bits": 16, "min": -32768, "max": 32767},
    tf.int32: {"bits": 32, "min": -2147483648, "max": 2147483647},
    tf.int64: {"bits": 64, "min": -9223372036854775808, "max": 9223372036854775807},
    tf.uint8: {"bits": 8, "min": 0, "max": 255},
    tf.uint16: {"bits": 16, "min": 0, "max": 65535},
    tf.uint32: {"bits": 32, "min": 0, "max": 4294967295},
    tf.uint64: {"bits": 64, "min": 0, "max": 18446744073709551615},
}


def _python_scalar_dtype(x: complex) -> DType | None:
    if isinstance(x, py_bool):
        return tf.bool
    if isinstance(x, int):
        if _iinfo[tf.int32]["min"] <= x <= _iinfo[tf.int32]["max"]:
            return tf.int32
        if _iinfo[tf.int64]["min"] <= x <= _iinfo[tf.int64]["max"]:
            return tf.int64
        if 0 <= x <= _iinfo[tf.uint64]["max"]:
            return tf.uint64
        return tf.int64
    if isinstance(x, float):
        if math.isfinite(x) and py_abs(x) > _finfo[tf.float32]["max"]:
            return tf.float64
        return tf.float32
    if isinstance(x, complex):
        return tf.complex128
    return None


def _as_dtype(x: Array | DType | complex) -> DType:
    if isinstance(x, tf.DType):
        return x
    if isinstance(x, tf.Tensor):
        return x.dtype
    dtype = _python_scalar_dtype(x)
    if dtype is not None:
        return dtype
    return tf.convert_to_tensor(x).dtype


def _real_dtype_for(dtype: DType) -> DType:
    dtype = tf.as_dtype(dtype)
    if dtype == tf.complex64:
        return tf.float32
    if dtype == tf.complex128:
        return tf.float64
    return dtype


def _is_integer(dtype: DType) -> py_bool:
    return dtype in _integral_dtypes


def _is_real(dtype: DType) -> py_bool:
    return dtype in _real_floating_dtypes


def _is_complex(dtype: DType) -> py_bool:
    return dtype in _complex_floating_dtypes


def _is_numeric(dtype: DType) -> py_bool:
    return dtype in _numeric_dtypes


def _accumulation_dtype(dtype: DType) -> DType:
    if dtype in (tf.uint8, tf.uint16, tf.uint32):
        return tf.uint32
    if dtype == tf.uint64:
        return tf.uint64
    if dtype in (tf.int8, tf.int16, tf.int32):
        return tf.int32
    if dtype == tf.int64:
        return tf.int64
    return dtype


def _normalize_axis(axis: int, ndim: int) -> int:
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise IndexError(f"axis {axis} is out of bounds for array of dimension {ndim}")
    return axis


def _normalize_axes(axis: int | Sequence[int] | None, ndim: int) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, int):
        axis = (axis,)
    axes = tuple(_normalize_axis(a, ndim) for a in axis)
    if len(set(axes)) != len(axes):
        raise ValueError("repeated axis")
    return axes


def _shape_tuple(x: Array) -> tuple[int, ...]:
    return tuple(x.shape.as_list())


def _dtype_of(x: Array | DType | complex) -> DType:
    if isinstance(x, tf.DType):
        return x
    if isinstance(x, tf.Tensor):
        return x.dtype
    dtype = _python_scalar_dtype(x)
    if dtype is not None:
        return dtype
    return tf.convert_to_tensor(x).dtype


def _promote_signed_unsigned(signed: DType, unsigned: DType) -> DType:
    signed_bits = _dtype_bits[signed]
    unsigned_bits = _dtype_bits[unsigned]
    for dtype in (tf.int16, tf.int32, tf.int64):
        bits = _dtype_bits[dtype]
        if bits >= signed_bits and bits > unsigned_bits:
            return dtype
    return tf.float64


def _promote_dtypes(dtype1: DType, dtype2: DType) -> DType:
    dtype1 = tf.as_dtype(dtype1)
    dtype2 = tf.as_dtype(dtype2)
    if dtype1 == dtype2:
        return dtype1
    if dtype1 == tf.bool:
        return dtype2
    if dtype2 == tf.bool:
        return dtype1
    if dtype1 in _complex_floating_dtypes or dtype2 in _complex_floating_dtypes:
        bits = py_max(
            _dtype_bits[_real_dtype_for(dtype1)],
            _dtype_bits[_real_dtype_for(dtype2)],
        )
        return _complex_for_bits[bits]
    if dtype1 in _real_floating_dtypes or dtype2 in _real_floating_dtypes:
        bits = py_max(
            _dtype_bits[dtype1] if dtype1 in _real_floating_dtypes else 0,
            _dtype_bits[dtype2] if dtype2 in _real_floating_dtypes else 0,
        )
        return _float_for_bits[py_max(bits, 32)]
    if dtype1 in _signed_dtypes and dtype2 in _signed_dtypes:
        return dtype1 if _dtype_bits[dtype1] >= _dtype_bits[dtype2] else dtype2
    if dtype1 in _unsigned_dtypes and dtype2 in _unsigned_dtypes:
        return dtype1 if _dtype_bits[dtype1] >= _dtype_bits[dtype2] else dtype2
    if dtype1 in _signed_dtypes and dtype2 in _unsigned_dtypes:
        return _promote_signed_unsigned(dtype1, dtype2)
    if dtype1 in _unsigned_dtypes and dtype2 in _signed_dtypes:
        return _promote_signed_unsigned(dtype2, dtype1)
    raise TypeError(f"Cannot promote {dtype1!r} and {dtype2!r}")


def _result_type(x: Array | DType | complex, y: Array | DType | complex) -> DType:
    return _promote_dtypes(_dtype_of(x), _dtype_of(y))


def _promote_scalar(dtype: DType, scalar: complex) -> DType:
    if isinstance(scalar, py_bool):
        return dtype if dtype != tf.bool else tf.bool
    if isinstance(scalar, int):
        return dtype if dtype in _numeric_dtypes else _promote_dtypes(dtype, tf.int32)
    if isinstance(scalar, float):
        if dtype in _real_floating_dtypes | _complex_floating_dtypes:
            return dtype
        return _promote_dtypes(dtype, tf.float64)
    if isinstance(scalar, complex):
        if dtype in _complex_floating_dtypes:
            return dtype
        return _promote_dtypes(dtype, tf.complex128)
    return _promote_dtypes(dtype, _dtype_of(scalar))


def _iter_nested_scalars(obj):
    if isinstance(obj, tf.Tensor):
        return
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray, memoryview)):
        for item in obj:
            yield from _iter_nested_scalars(item)
    else:
        yield obj


def _infer_nested_dtype(obj) -> DType | None:
    dtypes = []
    for scalar in _iter_nested_scalars(obj):
        if scalar is None:
            return None
        dtypes.append(_dtype_of(scalar))
    if not dtypes:
        return None
    return _reduce(_promote_dtypes, dtypes)


def _coerce_scalar_to_dtype(obj, dtype: DType):
    if dtype == tf.bool:
        return py_bool(obj)
    if dtype in _integral_dtypes:
        return int(obj)
    if dtype in _real_floating_dtypes:
        return float(obj)
    if dtype in _complex_floating_dtypes:
        return complex(obj)
    return obj


def _coerce_nested_to_dtype(obj, dtype: DType):
    if isinstance(obj, tf.Tensor):
        return obj
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray, memoryview)):
        return [_coerce_nested_to_dtype(item, dtype) for item in obj]
    return _coerce_scalar_to_dtype(obj, dtype)


def result_type(*arrays_and_dtypes: Array | DType | complex) -> DType:
    if not arrays_and_dtypes:
        raise ValueError("At least one array or dtype must be provided")
    if py_all(isinstance(x, _py_scalars) for x in arrays_and_dtypes):
        raise ValueError("At least one array or dtype must be provided")
    scalars = []
    others = []
    for x in arrays_and_dtypes:
        if isinstance(x, _py_scalars):
            scalars.append(x)
        else:
            others.append(x)
    dtype = _dtype_of(others[0])
    for other in others[1:]:
        dtype = _promote_dtypes(dtype, _dtype_of(other))
    for scalar in scalars:
        dtype = _promote_scalar(dtype, scalar)
    return dtype


def _moveaxis_permutation(
    ndim: int,
    source: int | Sequence[int],
    destination: int | Sequence[int],
) -> list[int]:
    if isinstance(source, int):
        source = (source,)
    if isinstance(destination, int):
        destination = (destination,)
    if len(source) != len(destination):
        raise ValueError("`source` and `destination` arguments must have the same number of elements")

    source_ = tuple(_normalize_axis(axis, ndim) for axis in source)
    destination_ = tuple(_normalize_axis(axis, ndim) for axis in destination)
    if len(set(source_)) != len(source_) or len(set(destination_)) != len(destination_):
        raise ValueError("repeated axis")

    order = [axis for axis in range(ndim) if axis not in source_]
    for dest, src in sorted(zip(destination_, source_, strict=True)):
        order.insert(dest, src)
    return order


def _moveaxis(
    x: Array,
    source: int | Sequence[int],
    destination: int | Sequence[int],
) -> Array:
    return tf.transpose(x, _moveaxis_permutation(x.ndim, source, destination))


def can_cast(from_: DType | Array, to: DType, /) -> py_bool:
    from_dtype = _dtype_of(from_)
    to = tf.as_dtype(to)
    if from_dtype == to:
        return True
    if from_dtype == tf.bool:
        return to == tf.bool
    if from_dtype in _signed_dtypes:
        if to in _signed_dtypes:
            return _dtype_bits[from_dtype] <= _dtype_bits[to]
        if to in _real_floating_dtypes:
            return _dtype_bits[to] > _dtype_bits[from_dtype]
        if to in _complex_floating_dtypes:
            return _dtype_bits[to] // 2 > _dtype_bits[from_dtype]
        return False
    if from_dtype in _unsigned_dtypes:
        if to in _unsigned_dtypes:
            return _dtype_bits[from_dtype] <= _dtype_bits[to]
        if to in _signed_dtypes:
            return _dtype_bits[from_dtype] < _dtype_bits[to]
        if to in _real_floating_dtypes:
            return _dtype_bits[to] > _dtype_bits[from_dtype]
        if to in _complex_floating_dtypes:
            return _dtype_bits[to] // 2 > _dtype_bits[from_dtype]
        return False
    if from_dtype in _real_floating_dtypes:
        if to in _real_floating_dtypes:
            return _dtype_bits[from_dtype] <= _dtype_bits[to]
        if to in _complex_floating_dtypes:
            return _dtype_bits[from_dtype] <= _dtype_bits[to] // 2
        return False
    if from_dtype in _complex_floating_dtypes and to in _complex_floating_dtypes:
        return _dtype_bits[from_dtype] <= _dtype_bits[to]
    return False


def _astype(x: Array, dtype: DType, copy: py_bool = False) -> Array:
    if x.dtype == dtype:
        return tf.identity(x) if copy else x
    return tf.cast(x, dtype)


def _negative_zero(dtype: DType) -> Array:
    dtype = tf.as_dtype(dtype)
    if dtype == tf.float16:
        return tf.bitcast(tf.constant(0x8000, dtype=tf.uint16), tf.float16)
    if dtype == tf.bfloat16:
        return tf.bitcast(tf.constant(0x8000, dtype=tf.uint16), tf.bfloat16)
    if dtype == tf.float32:
        return tf.bitcast(tf.constant(0x80000000, dtype=tf.uint32), tf.float32)
    if dtype == tf.float64:
        return tf.bitcast(tf.constant(0x8000000000000000, dtype=tf.uint64), tf.float64)
    raise TypeError(f"{dtype!r} is not a real floating dtype")


def _signed_zero_like(x: Array) -> Array:
    neg_zero = tf.broadcast_to(_negative_zero(x.dtype), tf.shape(x))
    return tf.where(signbit(x), neg_zero, tf.zeros_like(x))


def _python_scalar_to_tensor(x: complex, dtype: DType | None) -> Array | None:
    if dtype is None:
        dtype = _python_scalar_dtype(x)
    if dtype is None:
        return None
    dtype = tf.as_dtype(dtype)
    if isinstance(x, float) and dtype in _real_floating_dtypes and x == 0.0:
        if math.copysign(1.0, x) < 0:
            return _negative_zero(dtype)
        return tf.zeros((), dtype=dtype)
    if isinstance(x, complex) and dtype in _complex_floating_dtypes:
        real_dtype = _real_dtype_for(dtype)
        real_part = _python_scalar_to_tensor(x.real, real_dtype)
        imag_part = _python_scalar_to_tensor(x.imag, real_dtype)
        if real_part is None:
            real_part = tf.convert_to_tensor(x.real, dtype=real_dtype)
        if imag_part is None:
            imag_part = tf.convert_to_tensor(x.imag, dtype=real_dtype)
        return tf.complex(real_part, imag_part)
    return None


def _to_tensor(x: Array | complex, dtype: DType | None = None) -> Array:
    if isinstance(x, tf.Tensor):
        if dtype is not None:
            return _astype(x, dtype)
        return x
    out = _python_scalar_to_tensor(x, dtype)
    if out is not None:
        return out
    return tf.convert_to_tensor(x, dtype=dtype)


def _promote_two(x1: Array | complex, x2: Array | complex) -> tuple[Array, Array]:
    dtype = result_type(x1, x2)
    return _to_tensor(x1, dtype), _to_tensor(x2, dtype)


def _two_arg(f):
    @_wraps(f)
    def _f(x1, x2, /, **kwargs):
        x1, x2 = _promote_two(x1, x2)
        return f(x1, x2, **kwargs)

    if _f.__doc__ is None:
        _f.__doc__ = f"""\
Array API compatibility wrapper for tensorflow.{f.__name__}.

See the corresponding TensorFlow documentation and/or the array API
specification for more details.

"""
    return _f


def _logical_two_arg(f):
    @_wraps(f)
    def _f(x1, x2, /, **kwargs):
        return f(tf.convert_to_tensor(x1, dtype=tf.bool),
                 tf.convert_to_tensor(x2, dtype=tf.bool), **kwargs)

    return _f


def _preserve_zero_unary(f):
    @_wraps(f)
    def _f(x: Array, /) -> Array:
        out = f(x)
        if x.dtype in _real_floating_dtypes | _complex_floating_dtypes:
            return tf.where(x == tf.zeros((), dtype=x.dtype), x, out)
        return out

    return _f


def _check_device(device: Device | None) -> None:
    if device is None:
        return
    with tf.device(device):
        tf.constant(0)


def asarray(
    obj: Array | complex | NestedSequence[complex] | SupportsBufferProtocol,
    /,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    copy: py_bool | None = None,
    **kwargs: Any,
) -> Array:
    if copy is False and not isinstance(obj, tf.Tensor):
        raise ValueError("Unable to avoid copy while creating a TensorFlow tensor")
    with tf.device(device):
        if isinstance(obj, tf.Tensor):
            same_dtype = dtype is None or obj.dtype == dtype
            same_device = device is None or tf.identity(obj).device == obj.device
            if copy is False and not (same_dtype and same_device):
                raise ValueError("Unable to avoid copy while creating a TensorFlow tensor")
            out = obj if same_dtype else tf.cast(obj, dtype)
            if device is not None:
                out = tf.identity(out)
            if copy is True:
                out = tf.identity(out)
            return out
        try:
            if dtype is None:
                dtype = _infer_nested_dtype(obj)
            out = _python_scalar_to_tensor(obj, dtype)
            if out is None:
                obj_ = _coerce_nested_to_dtype(obj, dtype) if dtype is not None else obj
                out = tf.convert_to_tensor(obj_, dtype=dtype, **kwargs)
        except (TypeError, ValueError):
            obj_ = _coerce_nested_to_dtype(list(obj), dtype) if dtype is not None else list(obj)
            out = tf.convert_to_tensor(obj_, dtype=dtype, **kwargs)
        return tf.identity(out) if copy is True else out


def astype(
    x: Array,
    dtype: DType,
    /,
    *,
    copy: py_bool = True,
    device: Device | None = None,
) -> Array:
    with tf.device(device):
        out = _astype(x, dtype, copy=copy)
        if device is not None and out.device != tf.constant(0).device:
            out = tf.identity(out)
        return out


def from_dlpack(
    x,
    /,
    *,
    device: Device | None = None,
    copy: py_bool | None = None,
) -> Array:
    with tf.device(device):
        if isinstance(x, tf.Tensor):
            if copy is False and device is not None and x.device != tf.constant(0).device:
                raise ValueError("Unable to avoid copy while moving TensorFlow tensor")
            out = tf.identity(x) if copy is True or device is not None else x
        else:
            capsule = x.__dlpack__() if hasattr(x, "__dlpack__") else x
            # TensorFlow exposes DLPack conversion only in tf.experimental in
            # current releases.
            out = tf.experimental.dlpack.from_dlpack(capsule)
            if device is not None:
                out = tf.identity(out)
            elif copy is True:
                out = tf.identity(out)
        return out


def arange(
    start: float,
    /,
    stop: float | None = None,
    step: float = 1,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    **kwargs: object,
) -> Array:
    del kwargs
    if stop is None:
        start, stop = 0, start
    with tf.device(device):
        if step > 0 and stop <= start or step < 0 and stop >= start:
            if dtype is None:
                if py_all(isinstance(i, int) for i in (start, stop, step)):
                    dtype = tf.int32
                else:
                    dtype = tf.float32
            return tf.zeros((0,), dtype=dtype)
        if dtype is None:
            return tf.range(start, stop, step)
        work_dtype = tf.float64 if dtype in (tf.float64, tf.complex128) else tf.float32
        if dtype in _integral_dtypes:
            work_dtype = tf.int64
        return tf.cast(tf.range(start, stop, step, dtype=work_dtype), dtype)


def empty(
    shape: int | tuple[int, ...],
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    **kwargs: object,
) -> Array:
    del kwargs
    if isinstance(shape, int):
        shape = (shape,)
    with tf.device(device):
        return tf.zeros(shape, dtype=dtype or tf.float32)


def empty_like(
    x: Array,
    /,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    **kwargs: object,
) -> Array:
    del kwargs
    with tf.device(device):
        return tf.zeros_like(x, dtype=dtype)


def eye(
    n_rows: int,
    n_cols: int | None = None,
    /,
    *,
    k: int = 0,
    dtype: DType | None = None,
    device: Device | None = None,
    **kwargs: object,
) -> Array:
    del kwargs
    if n_cols is None:
        n_cols = n_rows
    with tf.device(device):
        if k >= n_cols or k <= -n_rows:
            return tf.zeros((n_rows, n_cols), dtype=dtype or tf.float32)
        rows = tf.range(n_rows)[:, newaxis]
        cols = tf.range(n_cols)[newaxis, :]
        return tf.cast(cols - rows == k, dtype or tf.float32)


def full(
    shape: int | tuple[int, ...],
    fill_value: complex,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    **kwargs: object,
) -> Array:
    del kwargs
    if isinstance(shape, int):
        shape = (shape,)
    value = tf.convert_to_tensor(fill_value, dtype=dtype)
    with tf.device(device):
        return tf.fill(shape, value)


def full_like(
    x: Array,
    /,
    fill_value: complex,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    **kwargs: object,
) -> Array:
    del kwargs
    return full(_shape_tuple(x), fill_value, dtype=dtype or x.dtype, device=device)


def linspace(
    start: float,
    stop: float,
    /,
    num: int,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    endpoint: py_bool = True,
    **kwargs: object,
) -> Array:
    del kwargs
    with tf.device(device):
        if num == 0:
            return tf.zeros((0,), dtype=dtype or tf.float32)
        out_dtype = dtype or tf.float32
        work_dtype = out_dtype if out_dtype in _real_floating_dtypes | _complex_floating_dtypes else tf.float32
        start_ = tf.convert_to_tensor(start, dtype=work_dtype)
        stop_ = tf.convert_to_tensor(stop, dtype=work_dtype)
        if endpoint:
            out = tf.linspace(start_, stop_, num)
        else:
            out = tf.linspace(start_, stop_, num + 1)[:-1]
        return tf.cast(out, out_dtype)


def ones(
    shape: int | tuple[int, ...],
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    **kwargs: object,
) -> Array:
    del kwargs
    with tf.device(device):
        return tf.ones(shape, dtype=dtype or tf.float32)


def ones_like(
    x: Array,
    /,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    **kwargs: object,
) -> Array:
    del kwargs
    with tf.device(device):
        return tf.ones_like(x, dtype=dtype)


def zeros(
    shape: int | tuple[int, ...],
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    **kwargs: object,
) -> Array:
    del kwargs
    with tf.device(device):
        return tf.zeros(shape, dtype=dtype or tf.float32)


def zeros_like(
    x: Array,
    /,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    **kwargs: object,
) -> Array:
    del kwargs
    with tf.device(device):
        return tf.zeros_like(x, dtype=dtype)


def tril(x: Array, /, *, k: int = 0) -> Array:
    rows = tf.range(x.shape[-2])[:, newaxis]
    cols = tf.range(x.shape[-1])[newaxis, :]
    return tf.where(cols - rows <= k, x, tf.zeros((), dtype=x.dtype))


def triu(x: Array, /, *, k: int = 0) -> Array:
    rows = tf.range(x.shape[-2])[:, newaxis]
    cols = tf.range(x.shape[-1])[newaxis, :]
    return tf.where(cols - rows >= k, x, tf.zeros((), dtype=x.dtype))


def isdtype(
    dtype: DType,
    kind: DType | str | tuple[DType | str, ...],
    *,
    _tuple: py_bool = True,
) -> py_bool:
    dtype = tf.as_dtype(dtype)
    if isinstance(kind, tuple) and _tuple:
        return py_any(isdtype(dtype, k, _tuple=False) for k in kind)
    if isinstance(kind, str):
        if kind == "bool":
            return dtype in _bool_dtypes
        if kind == "signed integer":
            return dtype in _signed_dtypes
        if kind == "unsigned integer":
            return dtype in _unsigned_dtypes
        if kind == "integral":
            return dtype in _integral_dtypes
        if kind == "real floating":
            return dtype in _real_floating_dtypes
        if kind == "complex floating":
            return dtype in _complex_floating_dtypes
        if kind == "numeric":
            return dtype in _numeric_dtypes
        raise ValueError(f"Unrecognized data type kind: {kind!r}")
    return dtype == tf.as_dtype(kind)


class _FInfo:
    def __init__(self, dtype: DType):
        real_dtype = _real_dtype_for(dtype)
        info = _finfo[real_dtype]
        self.bits = info["bits"]
        self.eps = info["eps"]
        self.max = info["max"]
        self.min = info["min"]
        self.smallest_normal = info["smallest_normal"]
        self.dtype = real_dtype


class _IInfo:
    def __init__(self, dtype: DType):
        info = _iinfo[dtype]
        self.bits = info["bits"]
        self.max = info["max"]
        self.min = info["min"]
        self.dtype = dtype


def finfo(type_: DType | Array, /) -> _FInfo:
    return _FInfo(_as_dtype(type_))


def iinfo(type_: DType | Array, /) -> _IInfo:
    return _IInfo(_as_dtype(type_))


def abs(x: Array, /) -> Array:
    if x.dtype in _unsigned_dtypes or x.dtype == tf.bool:
        return tf.identity(x)
    return tf.abs(x)


acos = tf.acos
acosh = tf.acosh
asin = _preserve_zero_unary(tf.asin)
asinh = _preserve_zero_unary(tf.asinh)
atan = _preserve_zero_unary(tf.atan)
atan2 = _two_arg(tf.atan2)
atanh = _preserve_zero_unary(tf.atanh)
add = _two_arg(tf.add)
conj = tf.math.conj
cos = tf.cos
cosh = tf.cosh
divide = _two_arg(tf.divide)
equal = _two_arg(tf.equal)
exp = tf.exp
greater = _two_arg(tf.greater)
greater_equal = _two_arg(tf.greater_equal)
less = _two_arg(tf.less)
less_equal = _two_arg(tf.less_equal)
logical_and = _logical_two_arg(tf.logical_and)
logical_not = tf.logical_not
logical_or = _logical_two_arg(tf.logical_or)
logical_xor = _logical_two_arg(tf.math.logical_xor)
maximum = _two_arg(tf.maximum)
minimum = _two_arg(tf.minimum)
multiply = _two_arg(tf.multiply)
not_equal = _two_arg(tf.not_equal)
positive = tf.identity
sin = _preserve_zero_unary(tf.sin)
sinh = _preserve_zero_unary(tf.sinh)
square = tf.square
sqrt = tf.sqrt
subtract = _two_arg(tf.subtract)
tan = _preserve_zero_unary(tf.tan)


def expm1(x: Array, /) -> Array:
    out = tf.math.expm1(x)
    if _is_complex(x.dtype):
        real_part = tf.math.real(x)
        imag_part = tf.math.imag(x)
        plus_inf_real = tf.math.is_inf(real_part) & (real_part > 0)
        exp_out = tf.exp(x) - tf.cast(1, x.dtype)
        inf_real = tf.fill(tf.shape(real_part), tf.cast(math.inf, real_part.dtype))
        zero_imag_out = tf.complex(inf_real, imag_part)
        special = tf.where(imag_part == 0, zero_imag_out, exp_out)
        out = tf.where(plus_inf_real, special, out)
        minus_inf_real = tf.math.is_inf(real_part) & (real_part < 0)
        minus_inf_imag = tf.where(tf.math.is_nan(imag_part), tf.zeros_like(imag_part), _signed_zero_like(imag_part))
        minus_inf_out = tf.complex(-tf.ones_like(real_part), minus_inf_imag)
        out = tf.where(minus_inf_real, minus_inf_out, out)
        nan_real_zero_imag = tf.math.is_nan(real_part) & (imag_part == 0)
        out = tf.where(nan_real_zero_imag, tf.complex(real_part, imag_part), out)
        zero_out = tf.complex(tf.zeros_like(real_part), imag_part)
        return tf.where((real_part == 0) & (imag_part == 0), zero_out, out)
    return out


def tanh(x: Array, /) -> Array:
    out = _preserve_zero_unary(tf.math.tanh)(x)
    if _is_complex(x.dtype):
        real_part = tf.math.real(x)
        imag_part = tf.math.imag(x)
        inf_real = tf.math.is_inf(real_part)
        real_out = tf.where(real_part > 0, tf.ones_like(real_part), -tf.ones_like(real_part))
        imag_out = tf.where(tf.math.is_nan(imag_part), tf.zeros_like(imag_part), _signed_zero_like(imag_part))
        return tf.where(inf_real, tf.complex(real_out, imag_out), out)
    return out


def remainder(x1: Array | complex, x2: Array | complex, /) -> Array:
    x1, x2 = _promote_two(x1, x2)
    out = tf.math.floormod(x1, x2)
    if out.dtype in _real_floating_dtypes:
        signed_zero = tf.broadcast_to(_signed_zero_like(x2), tf.shape(out))
        return tf.where(out == 0, signed_zero, out)
    return out


def bitwise_and(x1: Array | complex, x2: Array | complex, /) -> Array:
    x1, x2 = _promote_two(x1, x2)
    if x1.dtype == tf.bool:
        return tf.logical_and(x1, x2)
    return tf.bitwise.bitwise_and(x1, x2)


def bitwise_left_shift(x1: Array | complex, x2: Array | complex, /) -> Array:
    x1, x2 = _promote_two(x1, x2)
    out = tf.bitwise.left_shift(x1, x2)
    return tf.where(x2 >= tf.cast(_dtype_bits[x1.dtype], x2.dtype), tf.zeros((), dtype=x1.dtype), out)


def bitwise_invert(x: Array, /) -> Array:
    if x.dtype == tf.bool:
        return tf.logical_not(x)
    return tf.bitwise.invert(x)


def bitwise_or(x1: Array | complex, x2: Array | complex, /) -> Array:
    x1, x2 = _promote_two(x1, x2)
    if x1.dtype == tf.bool:
        return tf.logical_or(x1, x2)
    return tf.bitwise.bitwise_or(x1, x2)


def bitwise_right_shift(x1: Array | complex, x2: Array | complex, /) -> Array:
    x1, x2 = _promote_two(x1, x2)
    return tf.bitwise.right_shift(x1, x2)


def bitwise_xor(x1: Array | complex, x2: Array | complex, /) -> Array:
    x1, x2 = _promote_two(x1, x2)
    if x1.dtype == tf.bool:
        return tf.math.logical_xor(x1, x2)
    return tf.bitwise.bitwise_xor(x1, x2)


def ceil(x: Array, /) -> Array:
    return tf.identity(x) if _is_integer(x.dtype) else tf.math.ceil(x)


def floor(x: Array, /) -> Array:
    return tf.identity(x) if _is_integer(x.dtype) else tf.math.floor(x)


def trunc(x: Array, /) -> Array:
    if _is_integer(x.dtype):
        return tf.identity(x)
    return tf.where(x < 0, tf.math.ceil(x), tf.math.floor(x))


def copysign(x1: Array | complex, x2: Array | complex, /) -> Array:
    x1, x2 = _promote_two(x1, x2)
    return tf.where(signbit(x2), -tf.abs(x1), tf.abs(x1))


def hypot(x1: Array | complex, x2: Array | complex, /) -> Array:
    x1, x2 = _promote_two(x1, x2)
    return tf.sqrt(tf.square(x1) + tf.square(x2))


def imag(x: Array, /) -> Array:
    if _is_complex(x.dtype):
        return tf.math.imag(x)
    return tf.zeros_like(x)


def isfinite(x: Array, /) -> Array:
    if _is_integer(x.dtype) or x.dtype == tf.bool:
        return tf.ones(_shape_tuple(x), dtype=tf.bool)
    if _is_complex(x.dtype):
        return tf.math.is_finite(tf.math.real(x)) & tf.math.is_finite(tf.math.imag(x))
    return tf.math.is_finite(x)


def isinf(x: Array, /) -> Array:
    if _is_integer(x.dtype) or x.dtype == tf.bool:
        return tf.zeros(_shape_tuple(x), dtype=tf.bool)
    if _is_complex(x.dtype):
        return tf.math.is_inf(tf.math.real(x)) | tf.math.is_inf(tf.math.imag(x))
    return tf.math.is_inf(x)


def isnan(x: Array, /) -> Array:
    if _is_integer(x.dtype) or x.dtype == tf.bool:
        return tf.zeros(_shape_tuple(x), dtype=tf.bool)
    if _is_complex(x.dtype):
        return tf.math.is_nan(tf.math.real(x)) | tf.math.is_nan(tf.math.imag(x))
    return tf.math.is_nan(x)


def _complex_log(x: Array) -> Array:
    return tf.complex(tf.math.log(tf.abs(x)), tf.atan2(tf.math.imag(x), tf.math.real(x)))


def log(x: Array, /) -> Array:
    if _is_complex(x.dtype):
        return _complex_log(x)
    return tf.math.log(x)


def floor_divide(x1: Array | complex, x2: Array | complex, /) -> Array:
    x1, x2 = _promote_two(x1, x2)
    return tf.math.floordiv(x1, x2)


def log1p(x: Array, /) -> Array:
    if _is_complex(x.dtype):
        return log(tf.cast(1, x.dtype) + x)
    return tf.math.log1p(x)


def log2(x: Array, /) -> Array:
    if _is_complex(x.dtype):
        return _complex_log(x) / tf.cast(math.log(2.0), x.dtype)
    return tf.math.log(x) / tf.cast(math.log(2.0), x.dtype)


def log10(x: Array, /) -> Array:
    if _is_complex(x.dtype):
        return _complex_log(x) / tf.cast(math.log(10.0), x.dtype)
    return tf.math.log(x) / tf.cast(math.log(10.0), x.dtype)


def negative(x: Array, /) -> Array:
    if x.dtype in _unsigned_dtypes:
        return tf.subtract(tf.zeros_like(x), x)
    return tf.negative(x)


def pow(x1: Array | complex, x2: Array | complex, /) -> Array:
    x1, x2 = _promote_two(x1, x2)
    if x1.dtype in _unsigned_dtypes:
        out = tf.pow(tf.cast(x1, tf.float64), tf.cast(x2, tf.float64))
        return tf.cast(out, x1.dtype)
    return tf.pow(x1, x2)


def logaddexp(x1: Array | complex, x2: Array | complex, /) -> Array:
    x1, x2 = _promote_two(x1, x2)
    shape = tf.broadcast_static_shape(x1.shape, x2.shape)
    x1 = tf.broadcast_to(x1, shape)
    x2 = tf.broadcast_to(x2, shape)
    return tf.reduce_logsumexp(tf.stack([x1, x2]), axis=0)


def nextafter(x1: Array | complex, x2: Array | complex, /) -> Array:
    x1, x2 = _promote_two(x1, x2)
    return tf.math.nextafter(x1, x2)


def real(x: Array, /) -> Array:
    return tf.math.real(x) if _is_complex(x.dtype) else tf.identity(x)


def reciprocal(x: Array, /) -> Array:
    return tf.math.reciprocal(x)


def round(x: Array, /, *, decimals: int = 0) -> Array:
    if _is_integer(x.dtype):
        return tf.identity(x)
    if _is_complex(x.dtype):
        return tf.complex(round(real(x), decimals=decimals),
                          round(imag(x), decimals=decimals))
    if decimals == 0:
        return tf.round(x)
    factor = tf.cast(10 ** decimals, x.dtype)
    return tf.round(x * factor) / factor


def sign(x: Array, /) -> Array:
    if x.dtype in _unsigned_dtypes:
        return tf.where(x == 0, tf.zeros_like(x), tf.ones_like(x))
    if x.dtype in _real_floating_dtypes:
        return tf.where(x == 0, tf.zeros((), dtype=x.dtype), tf.sign(x))
    return tf.sign(x)


def signbit(x: Array, /) -> Array:
    if x.dtype in _unsigned_dtypes or x.dtype == tf.bool:
        return tf.zeros(_shape_tuple(x), dtype=tf.bool)
    if x.dtype in _signed_dtypes:
        return x < 0
    if x.dtype == tf.float16:
        return tf.bitcast(x, tf.int16) < 0
    if x.dtype == tf.bfloat16:
        return tf.bitcast(x, tf.int16) < 0
    if x.dtype == tf.float32:
        return tf.bitcast(x, tf.int32) < 0
    if x.dtype == tf.float64:
        return tf.bitcast(x, tf.int64) < 0
    raise TypeError("signbit is only defined for real-valued arrays")


def clip(
    x: Array,
    /,
    min: float | Array | None = None,
    max: float | Array | None = None,
) -> Array:
    def convert_bound(bound: float | Array) -> Array:
        if isinstance(bound, tf.Tensor):
            return tf.cast(bound, x.dtype)
        return tf.convert_to_tensor(bound, dtype=x.dtype)

    if min is None and max is None:
        return tf.identity(x)
    out = x
    if min is not None:
        out = maximum(out, convert_bound(min))
    if max is not None:
        out = minimum(out, convert_bound(max))
    return out


def _as_bool(x: Array) -> Array:
    if x.dtype == tf.bool:
        return x
    return x != tf.zeros((), dtype=x.dtype)


def all(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: py_bool = False,
) -> Array:
    x = _as_bool(x)
    if axis == ():
        return x
    return tf.reduce_all(x, axis=axis, keepdims=keepdims)


def any(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: py_bool = False,
) -> Array:
    x = _as_bool(x)
    if axis == ():
        return x
    return tf.reduce_any(x, axis=axis, keepdims=keepdims)


def max(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: py_bool = False,
) -> Array:
    if axis == ():
        return tf.identity(x)
    return tf.reduce_max(x, axis=axis, keepdims=keepdims)


def min(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: py_bool = False,
) -> Array:
    if axis == ():
        return tf.identity(x)
    return tf.reduce_min(x, axis=axis, keepdims=keepdims)


def mean(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: py_bool = False,
) -> Array:
    if axis == ():
        return tf.identity(x)
    return tf.reduce_mean(x, axis=axis, keepdims=keepdims)


def prod(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype: DType | None = None,
    keepdims: py_bool = False,
) -> Array:
    if dtype is None:
        dtype = _accumulation_dtype(x.dtype)
    if dtype is not None:
        x = tf.cast(x, dtype)
    if axis == ():
        return tf.identity(x)
    return tf.reduce_prod(x, axis=axis, keepdims=keepdims)


def sum(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype: DType | None = None,
    keepdims: py_bool = False,
) -> Array:
    if dtype is None:
        dtype = _accumulation_dtype(x.dtype)
    if dtype is not None:
        x = tf.cast(x, dtype)
    if axis == ():
        return tf.identity(x)
    return tf.reduce_sum(x, axis=axis, keepdims=keepdims)


def _axis_size(x: Array, axis: int | tuple[int, ...] | None) -> int:
    if axis is None:
        return math.prod(_shape_tuple(x))
    axes = _normalize_axes(axis, x.ndim)
    return math.prod(x.shape[a] for a in axes)


def var(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    correction: float = 0.0,
    keepdims: py_bool = False,
) -> Array:
    if axis == ():
        return tf.zeros_like(x)
    mean_ = tf.reduce_mean(x, axis=axis, keepdims=True)
    sq = tf.math.real((x - mean_) * tf.math.conj(x - mean_))
    n = _axis_size(x, axis)
    out = tf.reduce_sum(sq, axis=axis, keepdims=keepdims) / tf.cast(n - correction, sq.dtype)
    return tf.cast(out, x.dtype) if _is_real(x.dtype) else out


def std(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    correction: float = 0.0,
    keepdims: py_bool = False,
) -> Array:
    return tf.sqrt(var(x, axis=axis, correction=correction, keepdims=keepdims))


def cumulative_sum(
    x: Array,
    /,
    *,
    axis: int | None = None,
    dtype: DType | None = None,
    include_initial: py_bool = False,
) -> Array:
    if axis is None:
        if x.ndim > 1:
            raise ValueError("axis must be specified in cumulative_sum for more than one dimension")
        axis = 0
    if dtype is None:
        dtype = _accumulation_dtype(x.dtype)
    if dtype is not None:
        x = tf.cast(x, dtype)
    out = tf.math.cumsum(x, axis=axis)
    if include_initial:
        shape = list(_shape_tuple(x))
        shape[axis] = 1
        out = tf.concat([tf.zeros(shape, dtype=out.dtype), out], axis=axis)
    return out


def cumulative_prod(
    x: Array,
    /,
    *,
    axis: int | None = None,
    dtype: DType | None = None,
    include_initial: py_bool = False,
) -> Array:
    if axis is None:
        if x.ndim > 1:
            raise ValueError("axis must be specified in cumulative_prod for more than one dimension")
        axis = 0
    if dtype is None:
        dtype = _accumulation_dtype(x.dtype)
    if dtype is not None:
        x = tf.cast(x, dtype)
    out = tf.math.cumprod(x, axis=axis)
    if include_initial:
        shape = list(_shape_tuple(x))
        shape[axis] = 1
        out = tf.concat([tf.ones(shape, dtype=out.dtype), out], axis=axis)
    return out


def diff(
    x: Array,
    /,
    *,
    axis: int = -1,
    n: int = 1,
    prepend: Array | None = None,
    append: Array | None = None,
) -> Array:
    if prepend is not None:
        x = tf.concat([prepend, x], axis=axis)
    if append is not None:
        x = tf.concat([x, append], axis=axis)
    axis = _normalize_axis(axis, x.ndim)
    for _ in range(n):
        begin1 = [0] * x.ndim
        begin2 = [0] * x.ndim
        size = list(_shape_tuple(x))
        size[axis] -= 1
        begin2[axis] = 1
        x = tf.slice(x, begin2, size) - tf.slice(x, begin1, size)
    return x


def argsort(
    x: Array,
    /,
    *,
    axis: int = -1,
    descending: py_bool = False,
    stable: py_bool = True,
) -> Array:
    direction = "DESCENDING" if descending else "ASCENDING"
    return tf.argsort(x, axis=axis, direction=direction, stable=stable)


def sort(
    x: Array,
    /,
    *,
    axis: int = -1,
    descending: py_bool = False,
    stable: py_bool = True,
) -> Array:
    del stable
    direction = "DESCENDING" if descending else "ASCENDING"
    return tf.sort(x, axis=axis, direction=direction)


def take(x: Array, indices: Array, /, *, axis: int | None = None) -> Array:
    if axis is None:
        x = tf.reshape(x, (-1,))
        axis = 0
    indices = tf.where(indices < 0, indices + x.shape[axis], indices)
    return tf.gather(x, indices, axis=axis)


def take_along_axis(x: Array, indices: Array, /, *, axis: int = -1) -> Array:
    axis = _normalize_axis(axis, x.ndim)
    indices = tf.convert_to_tensor(indices)
    if indices.ndim != x.ndim:
        raise ValueError("`indices` and `x` must have the same number of dimensions")
    indices = tf.where(indices < 0, indices + x.shape[axis], indices)

    index_shape = tf.shape(indices)
    coords = []
    for dim in range(x.ndim):
        if dim == axis:
            coords.append(indices)
            continue
        if x.shape[dim] == indices.shape[dim]:
            coord = tf.range(index_shape[dim], dtype=indices.dtype)
            coord_shape = [1] * x.ndim
            coord_shape[dim] = index_shape[dim]
            coord = tf.reshape(coord, coord_shape)
            coords.append(tf.broadcast_to(coord, index_shape))
        elif x.shape[dim] == 1 or indices.shape[dim] == 1:
            coords.append(tf.zeros(index_shape, dtype=indices.dtype))
        else:
            raise ValueError("`indices` and `x` dimensions must match or broadcast outside `axis`")

    return tf.gather_nd(x, tf.stack(coords, axis=-1))


def matmul(x1: Array, x2: Array, /) -> Array:
    x1, x2 = _promote_two(x1, x2)
    dtype = x1.dtype
    x1_is_1d = x1.ndim == 1
    x2_is_1d = x2.ndim == 1
    if x1_is_1d:
        x1 = x1[..., newaxis, :]
    if x2_is_1d:
        x2 = x2[..., newaxis]
    if dtype in (tf.int8, tf.int16, tf.uint8, tf.uint16, tf.uint32, tf.uint64):
        out = tf.cast(tf.linalg.matmul(tf.cast(x1, tf.int64), tf.cast(x2, tf.int64)), dtype)
    else:
        out = tf.linalg.matmul(x1, x2)
    if x1_is_1d and x2_is_1d:
        return tf.squeeze(out, axis=(-2, -1))
    if x1_is_1d:
        return tf.squeeze(out, axis=-2)
    if x2_is_1d:
        return tf.squeeze(out, axis=-1)
    return out


def matrix_transpose(x: Array, /) -> Array:
    if x.ndim < 2:
        raise ValueError("x must be at least 2-dimensional for matrix_transpose")
    return tf.linalg.matrix_transpose(x)


def tensordot(
    x1: Array,
    x2: Array,
    /,
    *,
    axes: int | tuple[Sequence[int], Sequence[int]] = 2,
) -> Array:
    x1, x2 = _promote_two(x1, x2)
    if axes == 0 or (
        isinstance(axes, tuple) and len(axes[0]) == 0 and len(axes[1]) == 0
    ):
        x1_ = tf.reshape(x1, _shape_tuple(x1) + (1,) * x2.ndim)
        x2_ = tf.reshape(x2, (1,) * x1.ndim + _shape_tuple(x2))
        return tf.multiply(x1_, x2_)
    dtype = x1.dtype
    if dtype in (tf.int8, tf.int16, tf.uint8, tf.uint16, tf.uint32, tf.uint64):
        return tf.cast(
            tf.tensordot(tf.cast(x1, tf.int64), tf.cast(x2, tf.int64), axes=axes),
            dtype,
        )
    return tf.tensordot(x1, x2, axes=axes)


def vecdot(x1: Array, x2: Array, /, *, axis: int = -1) -> Array:
    x1, x2 = _promote_two(x1, x2)
    if x1.shape[axis] != x2.shape[axis]:
        raise ValueError("x1 and x2 must have the same size along the given axis")
    x1 = _moveaxis(x1, axis, -1)
    x2 = _moveaxis(x2, axis, -1)
    shape = tf.broadcast_static_shape(x1.shape, x2.shape)
    x1 = tf.broadcast_to(x1, shape)
    x2 = tf.broadcast_to(x2, shape)
    return tf.reduce_sum(tf.math.conj(x1) * x2, axis=-1)


def broadcast_shapes(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    shape = tf.TensorShape(())
    for s in shapes:
        shape = tf.broadcast_static_shape(shape, tf.TensorShape(s))
    return tuple(shape.as_list())


def broadcast_arrays(*arrays: Array) -> tuple[Array, ...]:
    shape = tf.broadcast_static_shape(arrays[0].shape, arrays[1].shape) if len(arrays) > 1 else arrays[0].shape
    for x in arrays[2:]:
        shape = tf.broadcast_static_shape(shape, x.shape)
    return tuple(tf.broadcast_to(x, shape) for x in arrays)


broadcast_to = tf.broadcast_to


def concat(
    arrays: tuple[Array, ...] | list[Array],
    /,
    *,
    axis: int | None = 0,
) -> Array:
    dtype = result_type(*arrays)
    arrays = tuple(_astype(x, dtype) for x in arrays)
    if axis is None:
        arrays = tuple(tf.reshape(x, (-1,)) for x in arrays)
        axis = 0
    return tf.concat(arrays, axis=axis)


def expand_dims(x: Array, /, axis: int | tuple[int, ...]) -> Array:
    if isinstance(axis, int):
        if axis < -x.ndim - 1 or axis > x.ndim:
            raise IndexError(f"{axis=} out of bounds for x with ndim {x.ndim}")
        return tf.expand_dims(x, axis)
    y_ndim = x.ndim + len(axis)
    axes = tuple(a + y_ndim if a < 0 else a for a in axis)
    if len(set(axes)) != len(axes) or py_any(a < 0 or a >= y_ndim for a in axes):
        raise ValueError(f"{axis=} not allowed for x with ndim {x.ndim}")
    shape_it = iter(_shape_tuple(x))
    shape = [1 if i in axes else next(shape_it) for i in range(y_ndim)]
    return tf.reshape(x, shape)


def flip(x: Array, /, *, axis: int | tuple[int, ...] | None = None) -> Array:
    return tf.reverse(x, axis=_normalize_axes(axis, x.ndim))


def meshgrid(*arrays: Array, indexing: Literal["xy", "ij"] = "xy") -> tuple[Array, ...]:
    return tuple(tf.meshgrid(*arrays, indexing=indexing))


def moveaxis(
    x: Array,
    source: int | Sequence[int],
    destination: int | Sequence[int],
) -> Array:
    return _moveaxis(x, source, destination)


def permute_dims(x: Array, /, axes: tuple[int, ...]) -> Array:
    return tf.transpose(x, axes)


def repeat(x: Array, repeats: int | Array, /, *, axis: int | None = None) -> Array:
    if x.dtype == tf.uint16:
        return tf.cast(tf.repeat(tf.cast(x, tf.uint32), repeats, axis=axis), tf.uint16)
    return tf.repeat(x, repeats, axis=axis)


def reshape(x: Array, /, shape: tuple[int, ...], *, copy: py_bool | None = None) -> Array:
    if copy is True:
        x = tf.identity(x)
    elif copy is False:
        # TensorFlow tensors are immutable views from the user's perspective.
        pass
    return tf.reshape(x, shape)


def roll(
    x: Array,
    /,
    shift: int | tuple[int, ...],
    *,
    axis: int | tuple[int, ...] | None = None,
) -> Array:
    if axis is None:
        return tf.reshape(tf.roll(tf.reshape(x, (-1,)), shift, 0), _shape_tuple(x))
    return tf.roll(x, shift, axis)


def squeeze(x: Array, /, axis: int | tuple[int, ...]) -> Array:
    axes = _normalize_axes(axis, x.ndim)
    if not axes:
        return x
    if py_any(x.shape[a] != 1 for a in axes):
        raise ValueError("cannot squeeze an axis with size other than one")
    return tf.squeeze(x, axis=axis)


def stack(arrays: tuple[Array, ...] | list[Array], /, *, axis: int = 0) -> Array:
    dtype = result_type(*arrays)
    arrays = tuple(_astype(x, dtype) for x in arrays)
    return tf.stack(arrays, axis=axis)


def tile(x: Array, repetitions: tuple[int, ...], /) -> Array:
    repetitions = tuple(repetitions)
    if len(repetitions) < x.ndim:
        repetitions = (1,) * (x.ndim - len(repetitions)) + repetitions
    elif len(repetitions) > x.ndim:
        x = tf.reshape(x, (1,) * (len(repetitions) - x.ndim) + _shape_tuple(x))
    if x.dtype == tf.uint16:
        return tf.cast(tf.tile(tf.cast(x, tf.uint32), repetitions), tf.uint16)
    return tf.tile(x, repetitions)


def unstack(x: Array, /, *, axis: int = 0) -> tuple[Array, ...]:
    if x.ndim == 0:
        raise ValueError("Input array must be at least 1-d.")
    return tuple(tf.unstack(x, axis=axis))


def argmax(x: Array, /, *, axis: int | None = None, keepdims: py_bool = False) -> Array:
    if axis is None:
        out = tf.argmax(tf.reshape(x, (-1,)), axis=0, output_type=tf.int64)
        if keepdims:
            out = tf.reshape(out, [1] * x.ndim)
        return out
    out = tf.argmax(x, axis=axis, output_type=tf.int64)
    if keepdims and axis is not None:
        out = tf.expand_dims(out, axis)
    return out


def argmin(x: Array, /, *, axis: int | None = None, keepdims: py_bool = False) -> Array:
    if axis is None:
        out = tf.argmin(tf.reshape(x, (-1,)), axis=0, output_type=tf.int64)
        if keepdims:
            out = tf.reshape(out, [1] * x.ndim)
        return out
    out = tf.argmin(x, axis=axis, output_type=tf.int64)
    if keepdims and axis is not None:
        out = tf.expand_dims(out, axis)
    return out


def count_nonzero(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: py_bool = False,
) -> Array:
    return tf.math.count_nonzero(x, axis=axis, keepdims=keepdims, dtype=tf.int64)


def nonzero(x: Array, /) -> tuple[Array, ...]:
    if x.ndim == 0:
        raise ValueError("nonzero() does not support zero-dimensional arrays")
    indices = tf.where(x)
    return tuple(tf.unstack(tf.transpose(indices), axis=0))


def searchsorted(
    x1: Array,
    x2: Array | int | float,
    /,
    *,
    side: Literal["left", "right"] = "left",
    sorter: Array | None = None,
) -> Array:
    if sorter is not None:
        x1 = tf.gather(x1, sorter)
    x2 = _to_tensor(x2, x1.dtype)
    x2_shape = _shape_tuple(x2)
    out = tf.searchsorted(x1, tf.reshape(x2, (-1,)), side=side, out_type=tf.int64)
    return tf.reshape(out, x2_shape)


def where(condition: Array, x1: Array | complex, x2: Array | complex, /) -> Array:
    x1, x2 = _promote_two(x1, x2)
    return tf.where(condition, x1, x2)


from ..common._aliases import UniqueAllResult, UniqueCountsResult, UniqueInverseResult


def _sort_unique_values(values: Array) -> Array:
    if values.dtype == tf.bool:
        return tf.argsort(tf.cast(values, tf.int8), stable=True)
    if _is_complex(values.dtype):
        order = tf.argsort(tf.math.imag(values), stable=True)
        values = tf.gather(values, order)
        order2 = tf.argsort(tf.math.real(values), stable=True)
        return tf.gather(order, order2)
    return tf.argsort(values, stable=True)


def _unique(x: Array) -> tuple[Array, Array, Array, Array]:
    flat = tf.reshape(x, (-1,))
    flat_size = math.prod(_shape_tuple(x))
    if flat_size == 0:
        empty_int = tf.zeros((0,), dtype=tf.int64)
        return flat, empty_int, tf.reshape(empty_int, _shape_tuple(x)), empty_int

    positions = tf.range(flat_size, dtype=tf.int64)
    eq = tf.equal(tf.expand_dims(flat, 0), tf.expand_dims(flat, 1))
    if x.dtype in _real_floating_dtypes:
        both_nan = tf.math.is_nan(tf.expand_dims(flat, 0)) & tf.math.is_nan(tf.expand_dims(flat, 1))
        eq = eq | both_nan
    elif _is_complex(x.dtype):
        nan = tf.math.is_nan(tf.math.real(flat)) | tf.math.is_nan(tf.math.imag(flat))
        both_nan = tf.expand_dims(nan, 0) & tf.expand_dims(nan, 1)
        eq = eq | both_nan

    first_for_position = tf.reduce_min(tf.where(eq, positions[tf.newaxis, :], flat_size), axis=1)
    unique_mask = first_for_position == positions
    values_unsorted = tf.boolean_mask(flat, unique_mask)
    first_unsorted = tf.boolean_mask(positions, unique_mask)
    counts_unsorted = tf.reduce_sum(tf.cast(tf.gather(eq, first_unsorted), tf.int64), axis=1)

    inverse_unsorted = tf.argmax(
        tf.equal(tf.expand_dims(first_for_position, -1), tf.expand_dims(first_unsorted, 0)),
        axis=-1,
        output_type=tf.int64,
    )

    order = _sort_unique_values(values_unsorted)
    values = tf.gather(values_unsorted, order)
    first = tf.gather(first_unsorted, order)
    counts = tf.gather(counts_unsorted, order)
    rank = tf.scatter_nd(
        tf.expand_dims(tf.cast(order, tf.int64), -1),
        tf.range(tf.shape(order)[0], dtype=tf.int64),
        tf.shape(order, out_type=tf.int64),
    )
    inverse = tf.reshape(tf.gather(rank, inverse_unsorted), _shape_tuple(x))
    return values, first, inverse, counts


def unique_all(x: Array) -> UniqueAllResult:
    return UniqueAllResult(*_unique(x))


def unique_counts(x: Array) -> UniqueCountsResult:
    values, _, _, counts = _unique(x)
    return UniqueCountsResult(values, counts)


def unique_inverse(x: Array) -> UniqueInverseResult:
    values, _, inverse, _ = _unique(x)
    return UniqueInverseResult(values, inverse)


def unique_values(x: Array) -> Array:
    values, _, _, _ = _unique(x)
    return values


def isin(x1: Array | int, x2: Array | int, /, *, invert: py_bool = False) -> Array:
    x1, x2 = _promote_two(x1, x2)
    x2 = tf.reshape(x2, (-1,))
    out = tf.reduce_any(tf.equal(tf.expand_dims(x1, -1), x2), axis=-1)
    return tf.logical_not(out) if invert else out


__all__ = [
    "abs",
    "acos",
    "acosh",
    "add",
    "all",
    "any",
    "arange",
    "argmax",
    "argmin",
    "argsort",
    "asarray",
    "asin",
    "asinh",
    "astype",
    "atan",
    "atan2",
    "atanh",
    "bitwise_and",
    "bitwise_left_shift",
    "bitwise_invert",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "bool",
    "broadcast_arrays",
    "broadcast_shapes",
    "broadcast_to",
    "can_cast",
    "ceil",
    "clip",
    "concat",
    "conj",
    "copysign",
    "cos",
    "cosh",
    "count_nonzero",
    "cumulative_prod",
    "cumulative_sum",
    "diff",
    "divide",
    "e",
    "empty",
    "empty_like",
    "equal",
    "exp",
    "expand_dims",
    "expm1",
    "eye",
    "finfo",
    "flip",
    "floor",
    "floor_divide",
    "from_dlpack",
    "full",
    "full_like",
    "greater",
    "greater_equal",
    "hypot",
    "iinfo",
    "imag",
    "inf",
    "isdtype",
    "isfinite",
    "isinf",
    "isin",
    "isnan",
    "less",
    "less_equal",
    "linspace",
    "log",
    "log1p",
    "log2",
    "log10",
    "logaddexp",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "matmul",
    "matrix_transpose",
    "max",
    "maximum",
    "mean",
    "meshgrid",
    "min",
    "minimum",
    "moveaxis",
    "multiply",
    "nan",
    "negative",
    "newaxis",
    "nextafter",
    "nonzero",
    "not_equal",
    "ones",
    "ones_like",
    "permute_dims",
    "pi",
    "positive",
    "pow",
    "prod",
    "real",
    "reciprocal",
    "remainder",
    "repeat",
    "reshape",
    "result_type",
    "roll",
    "round",
    "searchsorted",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "sort",
    "sqrt",
    "square",
    "squeeze",
    "stack",
    "std",
    "subtract",
    "sum",
    "take",
    "take_along_axis",
    "tan",
    "tanh",
    "tensordot",
    "tile",
    "tril",
    "triu",
    "trunc",
    "unstack",
    "unique_all",
    "unique_counts",
    "unique_inverse",
    "unique_values",
    "var",
    "vecdot",
    "where",
    "zeros",
    "zeros_like",
]


def __dir__() -> list[str]:
    return __all__
