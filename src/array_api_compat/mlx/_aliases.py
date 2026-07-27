import mlx.core as mx

from ..common._typing import NestedSequence, SupportsBufferProtocol
from ._typing import Array, Device, DType


def asarray(
    obj: Array | complex | NestedSequence[complex] | SupportsBufferProtocol,
    /,
    *,
    dtype: None | DType = None,
    device: None | Device = None,  # No device or stream argument in MLX.
    copy: bool | None = None,
) -> Array:
    """
    Note on MLX Compatibility:
        MLX does not support Ellipsis or tuple assignment in 0D or 1D arrays,
        example:
            >>> import mlx.core as mx
            >>> a = mx.array([1, 2, 3])
            >>> a[...] = 1
    """
    if (
        copy is False
        and isinstance(obj, Array)
        and (dtype is None or dtype == obj.dtype)
        and device is None
    ):
        return obj
    return mx.asarray(obj, dtype=dtype, copy=copy)


def arange(
    start: int | float,
    /,
    stop: int | float | None = None,
    step: int | float = 1,
    *,
    dtype: None | DType = None,
    device: None | Device = None,
):
    """
    Note on MLX Compatibility:
        Native `mx.arange` rejects some Python `int` arguments above 2**31 - 1.
        This wrapper handles empty and single-element ranges directly; larger
        int64 and uint64 ranges remain unsupported by MLX.
    """
    if stop is None:
        stop = start
        start = 0 if isinstance(stop, int) else 0.0
    # MLX defaults to float32 even when all arguments are integers,
    # so we must explicitly set the dtype to int32 in that case.
    is_float = (
        isinstance(start, float) or isinstance(stop, float) or isinstance(step, float)
    )
    dtype = dtype if dtype is not None else (mx.float32 if is_float else mx.int32)
    # MLX may overflow while converting a large step before determining that the
    # result has at most one element (for example, arange(0, 1, 2**31) is [0]),
    # so construct these trivial ranges directly.
    if (step > 0 and start >= stop) or (step < 0 and start <= stop):
        return mx.zeros((0,), dtype=dtype, stream=device)
    if (step > 0 and step >= stop - start) or (step < 0 and step <= stop - start):
        return mx.full((1,), start, dtype=dtype, stream=device)
    return mx.arange(start, stop, step, dtype=dtype, stream=device)


def empty_like(
    x: Array,
    /,
    *,
    dtype: None | DType = None,
    device: None | Device = None,
):
    return mx.empty(x.shape, dtype=x.dtype if dtype is None else dtype, stream=device)


def eye(
    n_rows: int,
    n_cols: int | None = None,
    /,
    *,
    k: int = 0,
    dtype: None | DType = None,
    device: None | Device = None,
):
    n_cols = n_rows if n_cols is None else n_cols
    if n_rows == 0 or n_cols == 0 or k >= n_cols or k <= -n_rows:
        return mx.zeros((n_rows, n_cols), dtype=dtype, stream=device)
    # mx.eye uses GPU scatter, which does not support int64 or uint64.
    # Build from exact 0/1 values in int32, then cast to the requested dtype.
    is_int64 = False
    if dtype is not None:
        is_int64 = dtype in (mx.int64, mx.uint64)
    result = mx.eye(
        n_rows, n_cols, k, dtype=mx.int32 if is_int64 else dtype, stream=device
    )
    return result.astype(dtype=dtype) if is_int64 else result


def meshgrid(*arrays: Array, indexing: str = "xy") -> tuple[Array, ...]:
    return tuple(mx.meshgrid(*arrays, indexing=indexing))


def ones_like(
    x: Array,
    /,
    *,
    dtype: None | DType = None,
    device: None | Device = None,
):
    return mx.ones(x.shape, dtype=x.dtype if dtype is None else dtype, stream=device)


def zeros_like(
    x: Array,
    /,
    *,
    dtype: None | DType = None,
    device: None | Device = None,
):
    return mx.zeros(x.shape, dtype=x.dtype if dtype is None else dtype, stream=device)


__all__ = [
    "arange",
    "asarray",
    "empty_like",
    "eye",
    "meshgrid",
    "ones_like",
    "zeros_like",
]
