from __future__ import annotations

from dask.array.linalg import svd
from ...common import _linalg
from ..._internal import get_xp

# Exports
from dask.array.linalg import * # noqa: F403
from dask.array import trace, outer

# These functions are in both the main and linalg namespaces
from dask.array import matmul, tensordot
from ._aliases import matrix_transpose, vecdot

import dask.array as da

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Union, Tuple
    from ...common._typing import ndarray

# cupy.linalg doesn't have __all__. If it is added, replace this with
#
# from cupy.linalg import __all__ as linalg_all
_n = {}
exec('from dask.array.linalg import *', _n)
del _n['__builtins__']
linalg_all = list(_n)
del _n

EighResult = _linalg.EighResult
QRResult = _linalg.QRResult
SlogdetResult = _linalg.SlogdetResult
SVDResult = _linalg.SVDResult
qr = get_xp(da)(_linalg.qr)
cholesky = get_xp(da)(_linalg.cholesky)
matrix_rank = get_xp(da)(_linalg.matrix_rank)
matrix_norm = get_xp(da)(_linalg.matrix_norm)

def svdvals(x: ndarray) -> Union[ndarray, Tuple[ndarray, ...]]:
    # TODO: can't avoid computing U or V for dask
    _, s, _ =  svd(x)
    return s

vector_norm = get_xp(da)(_linalg.vector_norm)
diagonal = get_xp(da)(_linalg.diagonal)

__all__ = linalg_all + ["trace", "outer", "matmul", "tensordot",
                        "matrix_transpose", "vecdot", "EighResult",
                        "QRResult", "SlogdetResult", "SVDResult", "qr",
                        "cholesky", "matrix_rank", "matrix_norm", "svdvals",
                        "vector_norm", "diagonal"]

del get_xp
del da
del _linalg
