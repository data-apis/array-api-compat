from numpy.linalg import *
from numpy.linalg import __all__ as linalg_all

from ..common import _linalg
from .._internal import get_xp

import numpy as np

cross = get_xp(np)(_linalg.cross)
diagonal = get_xp(np)(_linalg.diagonal)
matmul = get_xp(np)(_linalg.matmul)
cholesky = get_xp(np)(_linalg.cholesky)
matrix_rank = get_xp(np)(_linalg.matrix_rank)
pinv = get_xp(np)(_linalg.pinv)
matrix_norm = get_xp(np)(_linalg.matrix_norm)
matrix_transpose = get_xp(np)(_linalg.matrix_transpose)
outer = get_xp(np)(_linalg.outer)
svdvals = get_xp(np)(_linalg.svdvals)
tensordot = get_xp(np)(_linalg.tensordot)
trace = get_xp(np)(_linalg.trace)
vecdot = get_xp(np)(_linalg.vecdot)
vector_norm = get_xp(np)(_linalg.vector_norm)

__all__ = linalg_all + _linalg.__all__

del get_xp
del np
del linalg_all
del _linalg
