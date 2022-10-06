from __future__ import annotations

__all__ = [
    "ndarray",
    "Device",
    "Dtype",
    "NestedSequence",
    "SupportsBufferProtocol",
]

import sys
from typing import (
    Any,
    Literal,
    Union,
    TYPE_CHECKING,
    TypeVar,
    Protocol,
)

from numpy import (
    ndarray,
    dtype,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float32,
    float64,
)

_T_co = TypeVar("_T_co", covariant=True)

class NestedSequence(Protocol[_T_co]):
    def __getitem__(self, key: int, /) -> _T_co | NestedSequence[_T_co]: ...
    def __len__(self, /) -> int: ...

Device = Literal["cpu"]
if TYPE_CHECKING or sys.version_info >= (3, 9):
    Dtype = dtype[Union[
        int8,
        int16,
        int32,
        int64,
        uint8,
        uint16,
        uint32,
        uint64,
        float32,
        float64,
    ]]
else:
    Dtype = dtype

SupportsBufferProtocol = Any
