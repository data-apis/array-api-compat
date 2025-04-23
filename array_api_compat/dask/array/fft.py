from dask.array.fft import * # noqa: F403
# dask.array.fft doesn't have __all__. If it is added, replace this with
# from dask.array.fft import __all__ as fft_all
_n: dict[str, object] = {}
exec('from dask.array.fft import *', _n)
fft_all = list(_n)
del _n

from ...common import _fft
from ..._internal import get_xp

import dask.array as da

fftfreq = get_xp(da)(_fft.fftfreq)
rfftfreq = get_xp(da)(_fft.rfftfreq)

__all__ = fft_all + ["fftfreq", "rfftfreq"]

def __dir__() -> list[str]:
    return __all__
