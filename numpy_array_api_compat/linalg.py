from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Literal, Optional, Tuple, Union
    from numpy import ndarray

import numpy as np
from numpy.core.numeric import normalize_axis_tuple

def matrix_norm(x: ndarray, /, *, keepdims: bool = False, ord: Optional[Union[int, float, Literal['fro', 'nuc']]] = 'fro') -> ndarray:
    return np.linalg.norm(x, axis=(-2, -1), keepdims=keepdims, ord=ord)

# this function is new in the array API spec. Unlike transpose, it only
# transposes the last two axes.
def matrix_transpose(x: ndarray, /) -> ndarray:
    if x.ndim < 2:
        raise ValueError("x must be at least 2-dimensional for matrix_transpose")
    return np.swapaxes(x, -1, -2)

# svdvals is not in NumPy (but it is in SciPy). It is equivalent to
# np.linalg.svd(compute_uv=False).
def svdvals(x: ndarray, /) -> Union[ndarray, Tuple[ndarray, ...]]:
    return np.linalg.svd(x, compute_uv=False)

# vecdot is not in NumPy
def vecdot(x1: ndarray, x2: ndarray, /, *, axis: int = -1) -> ndarray:
    ndim = max(x1.ndim, x2.ndim)
    x1_shape = (1,)*(ndim - x1.ndim) + tuple(x1.shape)
    x2_shape = (1,)*(ndim - x2.ndim) + tuple(x2.shape)
    if x1_shape[axis] != x2_shape[axis]:
        raise ValueError("x1 and x2 must have the same size along the given axis")

    x1_, x2_ = np.broadcast_arrays(x1, x2)
    x1_ = np.moveaxis(x1_, axis, -1)
    x2_ = np.moveaxis(x2_, axis, -1)

    res = x1_[..., None, :] @ x2_[..., None]
    return res[..., 0, 0]

def vector_norm(x: ndarray, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False, ord: Optional[Union[int, float]] = 2) -> ndarray:
    # np.linalg.norm tries to do a matrix norm whenever axis is a 2-tuple or
    # when axis=None and the input is 2-D, so to force a vector norm, we make
    # it so the input is 1-D (for axis=None), or reshape so that norm is done
    # on a single dimension.
    if axis is None:
        # Note: np.linalg.norm() doesn't handle 0-D arrays
        x = x.ravel()
        _axis = 0
    elif isinstance(axis, tuple):
        # Note: The axis argument supports any number of axes, whereas
        # np.linalg.norm() only supports a single axis for vector norm.
        normalized_axis = normalize_axis_tuple(axis, x.ndim)
        rest = tuple(i for i in range(x.ndim) if i not in normalized_axis)
        newshape = axis + rest
        x = np.transpose(x, newshape).reshape(
            (np.prod([x.shape[i] for i in axis], dtype=int), *[x.shape[i] for i in rest]))
        _axis = 0
    else:
        _axis = axis

    res = np.linalg.norm(x, axis=_axis, ord=ord)

    if keepdims:
        # We can't reuse np.linalg.norm(keepdims) because of the reshape hacks
        # above to avoid matrix norm logic.
        shape = list(x.shape)
        _axis = normalize_axis_tuple(range(x.ndim) if axis is None else axis, x.ndim)
        for i in _axis:
            shape[i] = 1
        res = np.reshape(res, tuple(shape))

    return res

from numpy.linalg import *
from numpy.linalg import __all__ as linalg_all

# These are in the main NumPy namespace but not in numpy.linalg
from numpy import cross, diagonal, matmul, outer, tensordot, trace

__all__ = linalg_all.copy()
__all__ += ['cross', 'diagonal', 'matmul', 'matrix_norm', 'matrix_transpose',
            'outer', 'svdvals', 'tensordot', 'trace', 'vecdot', 'vector_norm']
