import mlx.core as mx

from .._internal import clone_module, get_xp
from ..common import _fft

__all__ = clone_module("mlx.core.fft", globals())

# Wrap all FFT functions via common helpers (handles float32->complex64 upcasting)
fft = get_xp(mx)(_fft.fft)
ifft = get_xp(mx)(_fft.ifft)
fftn = get_xp(mx)(_fft.fftn)
ifftn = get_xp(mx)(_fft.ifftn)
rfft = get_xp(mx)(_fft.rfft)
irfft = get_xp(mx)(_fft.irfft)
rfftn = get_xp(mx)(_fft.rfftn)
irfftn = get_xp(mx)(_fft.irfftn)
hfft = get_xp(mx)(_fft.hfft)
ihfft = get_xp(mx)(_fft.ihfft)
fftfreq = get_xp(mx)(_fft.fftfreq)
rfftfreq = get_xp(mx)(_fft.rfftfreq)
fftshift = get_xp(mx)(_fft.fftshift)
ifftshift = get_xp(mx)(_fft.ifftshift)

__all__ = sorted(set(__all__) | set(_fft.__all__))


def __dir__() -> list[str]:
    return __all__
