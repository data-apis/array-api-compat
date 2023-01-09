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
def max(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> array:
    # https://github.com/pytorch/pytorch/issues/29137
    if axis == ():
        return x
    return torch.amax(x, axis, keepdims=keepdims)

def min(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> array:
    # https://github.com/pytorch/pytorch/issues/29137
    if axis == ():
        return x
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

def prod(x: array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, dtype: Optional[Dtype] = None, keepdims: bool = False) -> array:
    # torch.prod doesn't support multiple axes
    # (https://github.com/pytorch/pytorch/issues/56586).
    if isinstance(axis, tuple):
        axes = _normalize_axes(axis, x.ndim)
        if keepdims:
            for a in axes:
                x = torch.prod(x, a, dtype=dtype, keepdims=keepdims)
            return x
        else:
            for i, a in enumerate(axes):
                x = torch.prod(x, a - i, dtype=dtype, keepdims=keepdims)
            return x
    if axis is None:
        # torch doesn't support keepdims with axis=None
        # (https://github.com/pytorch/pytorch/issues/71209)
        res = torch.prod(x, dtype=dtype)
        if keepdims:
            res = res[(None,)*x.ndim]
        return res

    return torch.prod(x, axis, dtype=dtype, keepdims=keepdims)

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

__all__ = ['permute_dims', 'max', 'min', 'expand_dims', 'full']
