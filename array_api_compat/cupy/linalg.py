from cupy.linalg import *
# cupy.linalg doesn't have __all__. If it is added, replace this with
#
# from cupy.linalg import __all__ as linalg_all
_n = {}
exec('from cupy.linalg import *', _n)
del _n['__builtins__']
linalg_all = list(_n)
del _n

from ..common import _linalg
from .._internal import get_xp

import cupy as cp

cross = get_xp(cp)(_linalg.cross)
matmul = get_xp(cp)(_linalg.matmul)
outer = get_xp(cp)(_linalg.outer)
tensordot = get_xp(cp)(_linalg.tensordot)
EighResult = _linalg.EighResult
QRResult = _linalg.QRResult
SlogdetResult = _linalg.SlogdetResult
SVDResult = _linalg.SVDResult
eigh = get_xp(cp)(_linalg.eigh)
qr = get_xp(cp)(_linalg.qr)
slogdet = get_xp(cp)(_linalg.slogdet)
svd = get_xp(cp)(_linalg.svd)
cholesky = get_xp(cp)(_linalg.cholesky)
matrix_rank = get_xp(cp)(_linalg.matrix_rank)
pinv = get_xp(cp)(_linalg.pinv)
matrix_norm = get_xp(cp)(_linalg.matrix_norm)
matrix_transpose = get_xp(cp)(_linalg.matrix_transpose)
svdvals = get_xp(cp)(_linalg.svdvals)
vecdot = get_xp(cp)(_linalg.vecdot)
vector_norm = get_xp(cp)(_linalg.vector_norm)
diagonal = get_xp(cp)(_linalg.diagonal)
trace = get_xp(cp)(_linalg.trace)

__all__ = linalg_all + _linalg.__all__

del get_xp
del cp
del linalg_all
del _linalg
