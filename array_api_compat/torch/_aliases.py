from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional, Tuple, Union
    from ..common._typing import Device, Dtype

import torch
array = torch.Tensor

# Basic renames
permute_dims = torch.permute

# These wrappers are mostly based on the fact that pytorch uses 'dim' instead
# of 'axis'.

# torch.min and torch.max return a tuple and don't support multiple axes https://github.com/pytorch/pytorch/issues/58745
def max(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> array:
    # https://github.com/pytorch/pytorch/issues/29137
    if axis == ():
        return torch.clone(x)
    return torch.amax(x, axis, keepdims=keepdims)

def min(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> array:
    # https://github.com/pytorch/pytorch/issues/29137
    if axis == ():
        return torch.clone(x)
    return torch.amin(x, axis, keepdims=keepdims)

def _normalize_axes(axis, ndim):
    axes = []
    lower, upper = -ndim, ndim - 1
    for a in axis:
        if a < lower or a > upper:
            # Match torch error message (e.g., from sum())
            raise IndexError(f"Dimension out of range (expected to be in range of [{lower}, {upper}], but got {a}")
        if a < 0:
            a = a + ndim
        if a in axes:
            # Match torch error message but use IndexError instead of RuntimeError
            raise IndexError(f"dim {a} appears multiple times in the list of dims")
        axes.append(a)
    return sorted(axes)


def _apply_keepdims(x, ndim, keepdims):
    if keepdims:
        return x[(None,)*ndim]
    return x

def prod(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, dtype: Optional[Dtype] = None, keepdims: bool = False) -> array:
    # torch.prod doesn't support multiple axes
    # (https://github.com/pytorch/pytorch/issues/56586).
    x = torch.asarray(x)
    ndim = x.ndim
    if isinstance(axis, tuple):
        axes = _normalize_axes(axis, x.ndim)
        for i, a in enumerate(axes):
            if keepdims:
                x = torch.prod(x, a, dtype=dtype)
                x = torch.unsqueeze(x, a)
            else:
                x = torch.prod(x, a - i, dtype=dtype)
        return x
    if axis is None:
        # torch doesn't support keepdims with axis=None
        # (https://github.com/pytorch/pytorch/issues/71209)
        res = torch.prod(x, dtype=dtype)
        res = _apply_keepdims(res, ndim, keepdims)
        return res

    return torch.prod(x, axis, dtype=dtype, keepdims=keepdims)

def any(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> array:
    # torch.any doesn't support multiple axes
    # (https://github.com/pytorch/pytorch/issues/56586).
    x = torch.asarray(x)
    ndim = x.ndim
    if axis == ():
        return x.to(torch.bool)
    if isinstance(axis, tuple):
        axes = _normalize_axes(axis, x.ndim)
        for i, a in enumerate(axes):
            if keepdims:
                x = torch.any(x, a)
                x = torch.unsqueeze(x, a)
            else:
                x = torch.any(x, a - i)
        return x.to(torch.bool)
    if axis is None:
        # torch doesn't support keepdims with axis=None
        # (https://github.com/pytorch/pytorch/issues/71209)
        res = torch.any(x)
        res = _apply_keepdims(res, ndim, keepdims)
        return res.to(torch.bool)

    # torch.any doesn't return bool for uint8
    return torch.any(x, axis, keepdims=keepdims).to(torch.bool)

def all(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> array:
    # torch.all doesn't support multiple axes
    # (https://github.com/pytorch/pytorch/issues/56586).
    x = torch.asarray(x)
    ndim = x.ndim
    if axis == ():
        return x.to(torch.bool)
    if isinstance(axis, tuple):
        axes = _normalize_axes(axis, x.ndim)
        for i, a in enumerate(axes):
            if keepdims:
                x = torch.all(x, a)
                x = torch.unsqueeze(x, a)
            else:
                x = torch.all(x, a - i)
        return x.to(torch.bool)
    if axis is None:
        # torch doesn't support keepdims with axis=None
        # (https://github.com/pytorch/pytorch/issues/71209)
        res = torch.all(x)
        res = _apply_keepdims(res, ndim, keepdims)
        return res.to(torch.bool)

    # torch.all doesn't return bool for uint8
    return torch.all(x, axis, keepdims=keepdims).to(torch.bool)

def expand_dims(x: array, /, *, axis: int = 0) -> array:
    return torch.unsqueeze(x, axis)

def full(shape: Union[int, Tuple[int, ...]],
         fill_value: Union[bool, int, float, complex],
         *,
         dtype: Optional[Dtype] = None,
         device: Optional[Device] = None,
         **kwargs) -> array:
    if isinstance(shape, int):
        shape = (shape,)

    return torch.full(shape, fill_value, dtype=dtype, device=device, **kwargs)

def astype(x: array, dtype: Dtype, /, *, copy: bool = True) -> array:
    return x.to(dtype, copy=copy)

__all__ = ['permute_dims', 'max', 'min', 'prod', 'any', 'all', 'expand_dims',
           'full', 'astype']
