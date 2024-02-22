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

# Note: unlike np.linalg.solve, the array API solve() only accepts x2 as a
# vector when it is exactly 1-dimensional. All other cases treat x2 as a stack
# of matrices. The np.linalg.solve behavior of allowing stacks of both
# matrices and vectors is ambiguous c.f.
# https://github.com/numpy/numpy/issues/15349 and
# https://github.com/data-apis/array-api/issues/285.

# To workaround this, the below is the code from np.linalg.solve except
# only calling solve1 in the exactly 1D case.

# This code is here instead of in common because it is numpy specific. Also
# note that CuPy's solve() does not currently support broadcasting (see
# https://github.com/cupy/cupy/blob/main/cupy/cublas.py#L43).
def solve(x1: _np.ndarray, x2: _np.ndarray, /) -> _np.ndarray:
    try:
        from numpy.linalg._linalg import (
        _makearray, _assert_stacked_2d, _assert_stacked_square,
        _commonType, isComplexType, _raise_linalgerror_singular
        )
    except ImportError:
        from numpy.linalg.linalg import (
        _makearray, _assert_stacked_2d, _assert_stacked_square,
        _commonType, isComplexType, _raise_linalgerror_singular
        )
    from numpy.linalg import _umath_linalg

    x1, _ = _makearray(x1)
    _assert_stacked_2d(x1)
    _assert_stacked_square(x1)
    x2, wrap = _makearray(x2)
    t, result_t = _commonType(x1, x2)

    # This part is different from np.linalg.solve
    if x2.ndim == 1:
        gufunc = _umath_linalg.solve1
    else:
        gufunc = _umath_linalg.solve

    # This does nothing currently but is left in because it will be relevant
    # when complex dtype support is added to the spec in 2022.
    signature = 'DD->D' if isComplexType(t) else 'dd->d'
    with _np.errstate(call=_raise_linalgerror_singular, invalid='call',
                      over='ignore', divide='ignore', under='ignore'):
        r = gufunc(x1, x2, signature=signature)

    return wrap(r.astype(result_t, copy=False))

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
    "solve",
    "svd",
    "svdvals",
    "tensordot",
    "trace",
    "vecdot",
    "vector_norm",
]
