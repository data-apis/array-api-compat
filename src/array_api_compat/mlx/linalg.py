from __future__ import annotations

import mlx.core as mx
import mlx.core.linalg as mx_linalg

from .._internal import clone_module, get_xp
from ..common import _linalg
from ._aliases import matmul, matrix_transpose, tensordot, vecdot

__all__ = clone_module("mlx.core.linalg", globals())

EighResult = _linalg.EighResult
EigResult = _linalg.EigResult
QRResult = _linalg.QRResult
SlogdetResult = _linalg.SlogdetResult
SVDResult = _linalg.SVDResult

# Wrap spec-compliant versions of linalg functions via common helpers
# eigh only works on CPU in MLX; pass stream=mx.cpu explicitly
def eigh(x: mx.array, /) -> _linalg.EighResult:
    return EighResult(*mx_linalg.eigh(x, stream=mx.cpu))
qr = get_xp(mx)(_linalg.qr)
slogdet = get_xp(mx)(_linalg.slogdet)
svd = get_xp(mx)(_linalg.svd)
cholesky = get_xp(mx)(_linalg.cholesky)
matrix_rank = get_xp(mx)(_linalg.matrix_rank)
pinv = get_xp(mx)(_linalg.pinv)
matrix_norm = get_xp(mx)(_linalg.matrix_norm)
vector_norm = get_xp(mx)(_linalg.vector_norm)
svdvals = get_xp(mx)(_linalg.svdvals)
diagonal = get_xp(mx)(_linalg.diagonal)
trace = get_xp(mx)(_linalg.trace)
cross = get_xp(mx)(_linalg.cross)
outer = get_xp(mx)(_linalg.outer)

# MLX linalg.solve signature matches spec (no ambiguous stacked-vector
# behaviour like NumPy), so pass through directly.
solve = mx_linalg.solve

# MLX linalg has no eig/eigvals returning complex — mark as unsupported.
# Raise a clear error rather than silently giving wrong results.
def eig(x, /):
    raise NotImplementedError(
        "eig() is not yet supported for MLX. "
        "MLX linalg.eigh() is available for symmetric/Hermitian matrices."
    )


def eigvals(x, /):
    raise NotImplementedError(
        "eigvals() is not yet supported for MLX. "
        "MLX linalg.eigh() is available for symmetric/Hermitian matrices."
    )

_all = ["eig", "eigvals", "solve", "cross", "outer",
    "matrix_transpose", "matmul", "tensordot", "vecdot"]

__all__ = sorted(
    set(__all__)
    | set(_linalg.__all__)
    | set(_all)
)


def __dir__() -> list[str]:
    return __all__
