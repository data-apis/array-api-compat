from dask.array.fft import * # noqa: F403
from numpy.fft import __all__ as fft_all

from ...common import _fft
from ..._internal import get_xp

import dask.array as da

# fft = get_xp(da)(_fft.fft)
# ifft = get_xp(da)(_fft.ifft)
# fftn = get_xp(da)(_fft.fftn)
# ifftn = get_xp(da)(_fft.ifftn)
# rfft = get_xp(da)(_fft.rfft)
# irfft = get_xp(da)(_fft.irfft)
# rfftn = get_xp(da)(_fft.rfftn)
# irfftn = get_xp(da)(_fft.irfftn)
# hfft = get_xp(da)(_fft.hfft)
# ihfft = get_xp(da)(_fft.ihfft)
fftfreq = get_xp(da)(_fft.fftfreq)
rfftfreq = get_xp(da)(_fft.rfftfreq)
# fftshift = get_xp(da)(_fft.fftshift)
# ifftshift = get_xp(da)(_fft.ifftshift)

__all__ = fft_all + _fft.__all__

del get_xp
del da
del fft_all
del _fft
