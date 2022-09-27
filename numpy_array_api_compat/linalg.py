from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ._typing import Literal, Optional, Tuple, Union
    from numpy import ndarray

import numpy as np
from numpy.core.numeric import normalize_axis_tuple

def matrix_norm(x: ndarray, /, *, keepdims: bool = False, ord: Optional[Union[int, float, Literal['fro', 'nuc']]] = 'fro') -> ndarray:
    return np.linalg.norm(x, axis=(-2, -1), keepdims=keepdims, ord=ord)

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
__all__ = linalg_all.copy()
__all__ += ['matrix_norm', 'vector_norm']
