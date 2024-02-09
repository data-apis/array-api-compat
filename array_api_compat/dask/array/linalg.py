import dask.array as _da
from dask.array import (
    matmul,
    outer,
    tensordot,
    trace,
)
from dask.array.linalg import *  # noqa: F401, F403

from ..._internal import _get_all_public_members
from ._aliases import (
    EighResult,
    QRResult,
    SlogdetResult,
    SVDResult,
    cholesky,
    diagonal,
    matrix_norm,
    matrix_rank,
    matrix_transpose,
    qr,
    svdvals,
    vecdot,
    vector_norm,
)

__all__ = [
    "matmul",
    "outer",
    "tensordot",
    "trace",
]

__all__ += _get_all_public_members(_da.linalg)

__all__ += [
    "EighResult",
    "QRResult",
    "SVDResult",
    "SlogdetResult",
    "cholesky",
    "diagonal",
    "matrix_norm",
    "matrix_rank",
    "matrix_transpose",
    "qr",
    "svdvals",
    "vecdot",
    "vector_norm",
]
