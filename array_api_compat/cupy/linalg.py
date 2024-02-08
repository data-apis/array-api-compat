import cupy as _cp

from .._internal import _get_all_public_members

_cupy_linalg_all = _get_all_public_members(_cp.linalg)

for _name in _cupy_linalg_all:
    globals()[_name] = getattr(_cp.linalg, _name)

from ._aliases import (  # noqa: E402
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
    "EighResult",
    "QRResult",
    "SVDResult",
    "SlogdetResult",
    "cholesky",
    "cross",
    "diagonal",
    "eigh",
    "matmul",
    "matrix_norm",
    "matrix_rank",
    "matrix_transpose",
    "outer",
    "pinv",
    "qr",
    "slogdet",
    "svd",
    "svdvals",
    "tensordot",
    "trace",
    "vecdot",
    "vector_norm",
]
