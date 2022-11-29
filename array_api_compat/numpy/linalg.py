from numpy.linalg import *
from numpy.linalg import __all__ as linalg_all

from ..common import _linalg
from .._internal import get_xp

import numpy as np

cross = get_xp(np)(_linalg.cross)
matmul = get_xp(np)(_linalg.matmul)
outer = get_xp(np)(_linalg.outer)
tensordot = get_xp(np)(_linalg.tensordot)
EighResult = _linalg.EighResult
QRResult = _linalg.QRResult
SlogdetResult = _linalg.SlogdetResult
SVDResult = _linalg.SVDResult
eigh = get_xp(np)(_linalg.eigh)
qr = get_xp(np)(_linalg.qr)
slogdet = get_xp(np)(_linalg.slogdet)
svd = get_xp(np)(_linalg.svd)
cholesky = get_xp(np)(_linalg.cholesky)
matrix_rank = get_xp(np)(_linalg.matrix_rank)
pinv = get_xp(np)(_linalg.pinv)
matrix_norm = get_xp(np)(_linalg.matrix_norm)
matrix_transpose = get_xp(np)(_linalg.matrix_transpose)
svdvals = get_xp(np)(_linalg.svdvals)
vecdot = get_xp(np)(_linalg.vecdot)
vector_norm = get_xp(np)(_linalg.vector_norm)
diagonal = get_xp(np)(_linalg.diagonal)
trace = get_xp(np)(_linalg.trace)

__all__ = linalg_all + _linalg.__all__

del get_xp
del np
del linalg_all
del _linalg
