from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional, Tuple, Union
    from ..common._typing import Device, Dtype

import torch
array = torch.Tensor

# Basic renames
permute_dims = torch.permute

def expand_dims(x: array, /, *, axis: int = 0) -> array:
    if axis > x.ndim or axis < -x.ndim - 1:
        raise IndexError("axis must be in the range [-x.ndim-1, x.ndim]")
    if axis < 0:
        axis = x.ndim + axis + 1
    slices = (slice(None),)*axis
    return x[slices + (None,)]

def full(shape: Union[int, Tuple[int, ...]],
         fill_value: Union[bool, int, float, complex],
         *,
         dtype: Optional[Dtype] = None,
         device: Optional[Device] = None,
         **kwargs) -> array:
    if isinstance(shape, int):
        shape = (shape,)

    return torch.full(shape, fill_value, dtype=dtype, device=device, **kwargs)

__all__ = ['permute_dims', 'expand_dims', 'full']
