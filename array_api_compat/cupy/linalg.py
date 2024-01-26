import cupy as cp
from .._internal import _get_all_public_members

_cupy_linalg_all = _get_all_public_members(cp.linalg)

for name in _cupy_linalg_all:
    globals()[name] = getattr(cp.linalg, name)

from ._aliases import ( # noqa: E402
    EighResult,
    QRResult,
    SlogdetResult,
    SVDResult,
    cholesky,
    cross,
    diagonal,
    eigh,
    matmul,
    matrix_norm,
    matrix_rank,
    matrix_transpose,
    outer,
    pinv,
    qr,
    slogdet,
    svd,
    svdvals,
    tensordot,
    trace,
    vecdot,
    vector_norm,
)

__all__ = []

__all__ += _cupy_linalg_all

__all__ += [
    "cross",
    "matmul",
    "outer",
    "tensordot",
    "EighResult",
    "QRResult",
    "SlogdetResult",
    "SVDResult",
    "eigh",
    "qr",
    "slogdet",
    "svd",
    "cholesky",
    "matrix_rank",
    "pinv",
    "matrix_norm",
    "matrix_transpose",
    "svdvals",
    "vecdot",
    "vector_norm",
    "diagonal",
    "trace",
]
