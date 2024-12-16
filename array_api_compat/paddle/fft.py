from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import paddle
    from ..common._typing import Device

    array = paddle.Tensor
    from typing import Optional, Union, Sequence, Literal

from paddle.fft import *  # noqa: F403
import paddle.fft


def fftn(
    x: array,
    /,
    *,
    s: Sequence[int] = None,
    axes: Sequence[int] = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
    **kwargs,
) -> array:
    return paddle.fft.fftn(x, s=s, axes=axes, norm=norm, **kwargs)


def ifftn(
    x: array,
    /,
    *,
    s: Sequence[int] = None,
    axes: Sequence[int] = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
    **kwargs,
) -> array:
    return paddle.fft.ifftn(x, s=s, axes=axes, norm=norm, **kwargs)


def rfftn(
    x: array,
    /,
    *,
    s: Sequence[int] = None,
    axes: Sequence[int] = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
    **kwargs,
) -> array:
    return paddle.fft.rfftn(x, s=s, axes=axes, norm=norm, **kwargs)


def irfftn(
    x: array,
    /,
    *,
    s: Sequence[int] = None,
    axes: Sequence[int] = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
    **kwargs,
) -> array:
    return paddle.fft.irfftn(x, s=s, axes=axes, norm=norm, **kwargs)


def fftshift(
    x: array,
    /,
    *,
    axes: Union[int, Sequence[int]] = None,
    **kwargs,
) -> array:
    return paddle.fft.fftshift(x, axes=axes, **kwargs)


def ifftshift(
    x: array,
    /,
    *,
    axes: Union[int, Sequence[int]] = None,
    **kwargs,
) -> array:
    return paddle.fft.ifftshift(x, axes=axes, **kwargs)


def fftfreq(
    n: int,
    /,
    *,
    d: float = 1.0,
    device: Optional[Device] = None,
) -> array:
    out = paddle.fft.fftfreq(n, d)
    if device is not None:
        out = out.to(device)
    return out


def rfftfreq(
    n: int,
    /,
    *,
    d: float = 1.0,
    device: Optional[Device] = None,
) -> array:
    out = paddle.fft.rfftfreq(n, d)
    if device is not None:
        out = out.to(device)
    return out


__all__ = paddle.fft.__all__ + [
    "fftn",
    "ifftn",
    "rfftn",
    "irfftn",
    "fftshift",
    "ifftshift",
    "fftfreq",
    "rfftfreq",
]

_all_ignore = ["paddle"]
