from numpy.linalg import *
from numpy.linalg import __all__ as linalg_all

from ..common import linalg
from .._internal import get_xp

import numpy as np

cross = get_xp(np)(linalg.cross)
diagonal = get_xp(np)(linalg.diagonal)
matmul = get_xp(np)(linalg.matmul)
cholesky = get_xp(np)(linalg.cholesky)
matrix_rank = get_xp(np)(linalg.matrix_rank)
pinv = get_xp(np)(linalg.pinv)
matrix_norm = get_xp(np)(linalg.matrix_norm)
matrix_transpose = get_xp(np)(linalg.matrix_transpose)
outer = get_xp(np)(linalg.outer)
svdvals = get_xp(np)(linalg.svdvals)
tensordot = get_xp(np)(linalg.tensordot)
trace = get_xp(np)(linalg.trace)
vecdot = get_xp(np)(linalg.vecdot)
vector_norm = get_xp(np)(linalg.vector_norm)

__all__ = linalg_all + linalg.__all__

del get_xp
del np
del linalg_all
