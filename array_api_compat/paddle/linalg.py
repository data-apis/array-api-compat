from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import paddle

    array = paddle.Tensor
    from paddle import dtype as Dtype
    from typing import Optional, Union, Tuple, Literal

    inf = float("inf")

from ._aliases import _fix_promotion, sum
from collections import namedtuple

import paddle
from paddle.linalg import *  # noqa: F403

# paddle.linalg doesn't define __all__
# from paddle.linalg import __all__ as linalg_all
from paddle import linalg as paddle_linalg

linalg_all = [i for i in dir(paddle_linalg) if not i.startswith("_")]

# outer is implemented in paddle but aren't in the linalg namespace
from paddle import outer
import paddle

# These functions are in both the main and linalg namespaces
from ._aliases import matmul, matrix_transpose, tensordot

# Note: paddle.linalg.cross does not default to axis=-1 (it defaults to the
# first axis with size 3)


# paddle.cross also does not support broadcasting when it would add new
# dimensions
def cross(x1: array, x2: array, /, *, axis: int = -1) -> array:
    x1, x2 = _fix_promotion(x1, x2, only_scalar=False)
    if not (-min(x1.ndim, x2.ndim) <= axis < max(x1.ndim, x2.ndim)):
        raise ValueError(f"axis {axis} out of bounds for cross product of arrays with shapes {x1.shape} and {x2.shape}")

    if not (x1.shape[axis] == x2.shape[axis] == 3):
        raise ValueError(f"cross product axis must have size 3, got {x1.shape[axis]} and {x2.shape[axis]}")

    x1, x2 = paddle.broadcast_tensors([x1, x2])
    return paddle_linalg.cross(x1, x2, axis=axis)


def vecdot(x1: array, x2: array, /, *, axis: int = -1, **kwargs) -> array:
    from ._aliases import isdtype

    x1, x2 = _fix_promotion(x1, x2, only_scalar=False)

    # paddle.linalg.vecdot incorrectly allows broadcasting along the contracted dimension
    if x1.shape[axis] != x2.shape[axis]:
        raise ValueError("x1 and x2 must have the same size along the given axis")

    # paddle.linalg.vecdot doesn't support integer dtypes
    if isdtype(x1.dtype, "integral") or isdtype(x2.dtype, "integral"):
        if kwargs:
            raise RuntimeError("vecdot kwargs not supported for integral dtypes")

        x1_ = paddle.moveaxis(x1, axis, -1)
        x2_ = paddle.moveaxis(x2, axis, -1)
        x1_, x2_ = paddle.broadcast_tensors([x1_, x2_])

        res = x1_[..., None, :] @ x2_[..., None]
        return res[..., 0, 0]
    return paddle.linalg.vecdot(x1, x2, axis=axis, **kwargs)


def solve(x1: array, x2: array, /, **kwargs) -> array:
    x1, x2 = _fix_promotion(x1, x2, only_scalar=False)

    if x2.ndim != 1 and x1.ndim - 1 == x2.ndim and x1.shape[:-1] == x2.shape:
        x2 = x2[None]
    return paddle.linalg.solve(x1, x2, **kwargs)


# paddle.trace doesn't support the offset argument and doesn't support stacking
def trace(x: array, /, *, offset: int = 0, dtype: Optional[Dtype] = None) -> array:
    # Use our wrapped sum to make sure it does upcasting correctly
    return sum(paddle.diagonal(x, offset=offset, axis1=-2, axis2=-1), axis=-1, dtype=dtype)

def diagonal(x: ndarray, / , *, offset: int = 0, **kwargs) -> ndarray:
    return paddle.diagonal(x, offset=offset, axis1=-2, axis2=-1, **kwargs)

def vector_norm(
    x: array,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
    ord: Union[int, float, Literal[inf, -inf]] = 2,
    **kwargs,
) -> array:
    # paddle.vector_norm incorrectly treats axis=() the same as axis=None
    if axis == ():
        out = kwargs.get("out")
        if out is None:
            dtype = None
            if x.dtype == paddle.complex64:
                dtype = paddle.float32
            elif x.dtype == paddle.complex128:
                dtype = paddle.float64

            out = paddle.zeros_like(x, dtype=dtype)

        # The norm of a single scalar works out to abs(x) in every case except
        # for ord=0, which is x != 0.
        if ord == 0:
            out[:] = x != 0
        else:
            out[:] = paddle.abs(x)
        return out
    return paddle.linalg.vector_norm(x, p=ord, axis=axis, keepdim=keepdims, **kwargs)


def matrix_norm(
    x: array,
    /,
    *,
    keepdims: bool = False,
    ord: Optional[Union[int, float, Literal["fro", "nuc"]]] = "fro",
) -> array:
    res = paddle.linalg.matrix_norm(x, p=ord, axis=(-2, -1), keepdim=keepdims)
    if res.dtype == paddle.complex64 :
        res = paddle.cast(res, "float32")
    if res.dtype == paddle.complex128:
        res = paddle.cast(res, "float64")
    return res

def pinv(x: array, /, *, rtol: Optional[Union[float, array]] = None) -> array:
    if rtol is None:
        return paddle.linalg.pinv(x)
    
    # change rtol shape 
    if isinstance(rtol, (int, float)):
        rtol = paddle.to_tensor(rtol, dtype=x.dtype)
    
    # broadcast rtol to [..., 1] 
    if rtol.ndim > 0:
        rtol = rtol.unsqueeze(-1)  

    return paddle.linalg.pinv(x, rcond=rtol)


def slogdet(x: array):
    return tuple_to_namedtuple(paddle.linalg.slogdet(x), ["sign", "logabsdet"])

def tuple_to_namedtuple(data, fields):
    nt_class = namedtuple('DynamicNameTuple', fields)
    return nt_class(*data)

def eigh(x: array):
    return tuple_to_namedtuple(paddle.linalg.eigh(x), ['eigenvalues', 'eigenvectors'])

def qr(x: array, mode: Optional[str] = None) -> array:
    if mode is None:
        return tuple_to_namedtuple(paddle.linalg.qr(x), ['Q', 'R'])

    return tuple_to_namedtuple(paddle.linalg.qr(x, mode), ['Q', 'R'])


def svd(x: array, full_matrices: Optional[bool]= None) -> array:
    if full_matrices is None :
        return tuple_to_namedtuple(paddle.linalg.svd(x, full_matrices=True), ['U', 'S', 'Vh'])
    return tuple_to_namedtuple(paddle.linalg.svd(x, full_matrices), ['U', 'S', 'Vh'])

def svdvals(x: array) -> array:
    return paddle.linalg.svd(x)[1]

__all__ = linalg_all + [
    "outer",
    "matmul",
    "matrix_transpose",
    "matrix_norm",
    "tensordot",
    "cross",
    "vecdot",
    "solve",
    "trace",
    "vector_norm",
    "slogdet",
    "eigh",
    "diagonal",
    "svdvals"
]

_all_ignore = ["paddle_linalg", "sum"]

del linalg_all
