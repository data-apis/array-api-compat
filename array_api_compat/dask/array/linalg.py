from __future__ import annotations

from dask.array.linalg import *
from ...common import _linalg
from ..._internal import get_xp
from dask.array import matmul, tensordot, trace, outer
from ._aliases import matrix_transpose, vecdot

import dask.array as da

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional, Union, Tuple
    from ...common._typing import ndarray, Device, Dtype

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

del get_xp
del da
del _linalg
