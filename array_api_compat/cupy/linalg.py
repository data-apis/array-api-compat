from cupy.linalg import *
# cupy.linalg doesn't have __all__. If it is added, replace this with
#
# from cupy.linalg import __all__ as linalg_all
_n = {}
exec('from cupy.linalg import *', _n)
del _n['__builtins__']
linalg_all = list(_n)
del _n

from ..common import linalg
from .._internal import get_xp

import cupy as cp

cross = get_xp(cp)(linalg.cross)
diagonal = get_xp(cp)(linalg.diagonal)
matmul = get_xp(cp)(linalg.matmul)
cholesky = get_xp(cp)(linalg.cholesky)
matrix_rank = get_xp(cp)(linalg.matrix_rank)
pinv = get_xp(cp)(linalg.pinv)
matrix_norm = get_xp(cp)(linalg.matrix_norm)
matrix_transpose = get_xp(cp)(linalg.matrix_transpose)
outer = get_xp(cp)(linalg.outer)
svdvals = get_xp(cp)(linalg.svdvals)
tensordot = get_xp(cp)(linalg.tensordot)
trace = get_xp(cp)(linalg.trace)
vecdot = get_xp(cp)(linalg.vecdot)
vector_norm = get_xp(cp)(linalg.vector_norm)

__all__ = linalg_all + linalg.__all__

del get_xp
del cp
del linalg_all
