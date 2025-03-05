from __future__ import annotations

from collections.abc import Sequence
from typing import Union, Optional, Literal

from ._typing import Device, Array, DType, Namespace

# Note: NumPy fft functions improperly upcast float32 and complex64 to
# complex128, which is why we require wrapping them all here.

def fft(
    x: Array,
    /,
    xp: Namespace,
    *,
    n: Optional[int] = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Array:
    res = xp.fft.fft(x, n=n, axis=axis, norm=norm)
    if x.dtype in [xp.float32, xp.complex64]:
        return res.astype(xp.complex64)
    return res

def ifft(
    x: Array,
    /,
    xp: Namespace,
    *,
    n: Optional[int] = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Array:
    res = xp.fft.ifft(x, n=n, axis=axis, norm=norm)
    if x.dtype in [xp.float32, xp.complex64]:
        return res.astype(xp.complex64)
    return res

def fftn(
    x: Array,
    /,
    xp: Namespace,
    *,
    s: Sequence[int] = None,
    axes: Sequence[int] = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Array:
    res = xp.fft.fftn(x, s=s, axes=axes, norm=norm)
    if x.dtype in [xp.float32, xp.complex64]:
        return res.astype(xp.complex64)
    return res

def ifftn(
    x: Array,
    /,
    xp: Namespace,
    *,
    s: Sequence[int] = None,
    axes: Sequence[int] = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Array:
    res = xp.fft.ifftn(x, s=s, axes=axes, norm=norm)
    if x.dtype in [xp.float32, xp.complex64]:
        return res.astype(xp.complex64)
    return res

def rfft(
    x: Array,
    /,
    xp: Namespace,
    *,
    n: Optional[int] = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Array:
    res = xp.fft.rfft(x, n=n, axis=axis, norm=norm)
    if x.dtype == xp.float32:
        return res.astype(xp.complex64)
    return res

def irfft(
    x: Array,
    /,
    xp: Namespace,
    *,
    n: Optional[int] = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Array:
    res = xp.fft.irfft(x, n=n, axis=axis, norm=norm)
    if x.dtype == xp.complex64:
        return res.astype(xp.float32)
    return res

def rfftn(
    x: Array,
    /,
    xp: Namespace,
    *,
    s: Sequence[int] = None,
    axes: Sequence[int] = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Array:
    res = xp.fft.rfftn(x, s=s, axes=axes, norm=norm)
    if x.dtype == xp.float32:
        return res.astype(xp.complex64)
    return res

def irfftn(
    x: Array,
    /,
    xp: Namespace,
    *,
    s: Sequence[int] = None,
    axes: Sequence[int] = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Array:
    res = xp.fft.irfftn(x, s=s, axes=axes, norm=norm)
    if x.dtype == xp.complex64:
        return res.astype(xp.float32)
    return res

def hfft(
    x: Array,
    /,
    xp: Namespace,
    *,
    n: Optional[int] = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Array:
    res = xp.fft.hfft(x, n=n, axis=axis, norm=norm)
    if x.dtype in [xp.float32, xp.complex64]:
        return res.astype(xp.float32)
    return res

def ihfft(
    x: Array,
    /,
    xp: Namespace,
    *,
    n: Optional[int] = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Array:
    res = xp.fft.ihfft(x, n=n, axis=axis, norm=norm)
    if x.dtype in [xp.float32, xp.complex64]:
        return res.astype(xp.complex64)
    return res

def fftfreq(
    n: int,
    /,
    xp: Namespace,
    *,
    d: float = 1.0,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> Array:
    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")
    res = xp.fft.fftfreq(n, d=d)
    if dtype is not None:
        return res.astype(dtype)
    return res

def rfftfreq(
    n: int,
    /,
    xp: Namespace,
    *,
    d: float = 1.0,
    dtype: Optional[DType] = None,
    device: Optional[Device] = None,
) -> Array:
    if device not in ["cpu", None]:
        raise ValueError(f"Unsupported device {device!r}")
    res = xp.fft.rfftfreq(n, d=d)
    if dtype is not None:
        return res.astype(dtype)
    return res

def fftshift(
    x: Array, /, xp: Namespace, *, axes: Union[int, Sequence[int]] = None
) -> Array:
    return xp.fft.fftshift(x, axes=axes)

def ifftshift(
    x: Array, /, xp: Namespace, *, axes: Union[int, Sequence[int]] = None
) -> Array:
    return xp.fft.ifftshift(x, axes=axes)

__all__ = [
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
]
