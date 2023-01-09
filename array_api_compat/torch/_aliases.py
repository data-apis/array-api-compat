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
