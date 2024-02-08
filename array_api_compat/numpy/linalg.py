import numpy as _np

from .._internal import _get_all_public_members

_numpy_linalg_all = _get_all_public_members(_np.linalg)

for _name in _numpy_linalg_all:
    globals()[_name] = getattr(_np.linalg, _name)


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

__all__ += _numpy_linalg_all

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
