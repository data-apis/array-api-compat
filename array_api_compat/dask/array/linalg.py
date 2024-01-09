from dask.array.linalg import *
from dask.array.linalg import __all__ as linalg_all

from ...common import _linalg
from ..._internal import get_xp
from ._aliases import (matmul, matrix_transpose, tensordot, vecdot)

import dask.array as da

cross = get_xp(da)(_linalg.cross)
outer = get_xp(da)(_linalg.outer)
EighResult = _linalg.EighResult
QRResult = _linalg.QRResult
SlogdetResult = _linalg.SlogdetResult
SVDResult = _linalg.SVDResult
eigh = get_xp(da)(_linalg.eigh)
qr = get_xp(da)(_linalg.qr)
slogdet = get_xp(da)(_linalg.slogdet)
svd = get_xp(da)(_linalg.svd)
cholesky = get_xp(da)(_linalg.cholesky)
matrix_rank = get_xp(da)(_linalg.matrix_rank)
pinv = get_xp(da)(_linalg.pinv)
matrix_norm = get_xp(da)(_linalg.matrix_norm)
svdvals = get_xp(da)(_linalg.svdvals)
vector_norm = get_xp(da)(_linalg.vector_norm)
diagonal = get_xp(da)(_linalg.diagonal)
trace = get_xp(da)(_linalg.trace)

__all__ = linalg_all + _linalg.__all__

del get_xp
del da
del linalg_all
del _linalg
