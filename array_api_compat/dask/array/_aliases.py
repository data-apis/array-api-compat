from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from ..._internal import get_xp
from ...common import _aliases, _linalg
from ...common._helpers import _check_device

if TYPE_CHECKING:
    from typing import Optional, Tuple, Union

    from ...common._typing import Device, Dtype, Array

import dask.array as da

isdtype = get_xp(np)(_aliases.isdtype)
astype = _aliases.astype

# Common aliases

# This arange func is modified from the common one to
# not pass stop/step as keyword arguments, which will cause
# an error with dask


# TODO: delete the xp stuff, it shouldn't be necessary
def dask_arange(
    start: Union[int, float],
    /,
    stop: Optional[Union[int, float]] = None,
    step: Union[int, float] = 1,
    *,
    xp,
    dtype: Optional[Dtype] = None,
    device: Optional[Device] = None,
    **kwargs,
) -> Array:
    _check_device(xp, device)
    args = [start]
    if stop is not None:
        args.append(stop)
    else:
        # stop is None, so start is actually stop
        # prepend the default value for start which is 0
        args.insert(0, 0)
    args.append(step)
    return xp.arange(*args, dtype=dtype, **kwargs)


arange = get_xp(da)(dask_arange)
eye = get_xp(da)(_aliases.eye)

asarray = partial(_aliases._asarray, namespace="dask.array")
asarray.__doc__ = _aliases._asarray.__doc__

linspace = get_xp(da)(_aliases.linspace)
eye = get_xp(da)(_aliases.eye)
UniqueAllResult = get_xp(da)(_aliases.UniqueAllResult)
UniqueCountsResult = get_xp(da)(_aliases.UniqueCountsResult)
UniqueInverseResult = get_xp(da)(_aliases.UniqueInverseResult)
unique_all = get_xp(da)(_aliases.unique_all)
unique_counts = get_xp(da)(_aliases.unique_counts)
unique_inverse = get_xp(da)(_aliases.unique_inverse)
unique_values = get_xp(da)(_aliases.unique_values)
permute_dims = get_xp(da)(_aliases.permute_dims)
std = get_xp(da)(_aliases.std)
var = get_xp(da)(_aliases.var)
empty = get_xp(da)(_aliases.empty)
empty_like = get_xp(da)(_aliases.empty_like)
full = get_xp(da)(_aliases.full)
full_like = get_xp(da)(_aliases.full_like)
ones = get_xp(da)(_aliases.ones)
ones_like = get_xp(da)(_aliases.ones_like)
zeros = get_xp(da)(_aliases.zeros)
zeros_like = get_xp(da)(_aliases.zeros_like)
reshape = get_xp(da)(_aliases.reshape)
matrix_transpose = get_xp(da)(_aliases.matrix_transpose)
vecdot = get_xp(da)(_aliases.vecdot)

nonzero = get_xp(da)(_aliases.nonzero)
sum = get_xp(np)(_aliases.sum)
prod = get_xp(np)(_aliases.prod)
ceil = get_xp(np)(_aliases.ceil)
floor = get_xp(np)(_aliases.floor)
trunc = get_xp(np)(_aliases.trunc)
matmul = get_xp(np)(_aliases.matmul)
tensordot = get_xp(np)(_aliases.tensordot)


EighResult = _linalg.EighResult
QRResult = _linalg.QRResult
SlogdetResult = _linalg.SlogdetResult
SVDResult = _linalg.SVDResult
qr = get_xp(da)(_linalg.qr)
cholesky = get_xp(da)(_linalg.cholesky)
matrix_rank = get_xp(da)(_linalg.matrix_rank)
matrix_norm = get_xp(da)(_linalg.matrix_norm)

# Wrap the svd functions to not pass full_matrices to dask
# when full_matrices=False (as that is the defualt behavior for dask),
# and dask doesn't have the full_matrices keyword
_svd = get_xp(da)(_linalg.svd)

def svd(x: Array, full_matrices: bool = True, **kwargs) -> SVDResult:
    if full_matrices:
        return _svd(x, full_matrices=full_matrices, **kwargs)
    return _svd(x, **kwargs)


def svdvals(x: Array) -> Array:
    # TODO: can't avoid computing U or V for dask
    _, s, _ = da.linalg.svd(x)
    return s


vector_norm = get_xp(da)(_linalg.vector_norm)
diagonal = get_xp(da)(_linalg.diagonal)
