from __future__ import annotations
from types import ModuleType as Namespace
from typing import Any, TypeVar, Protocol

__all__ = [
    "Array",
    "DType",
    "Device",
    "Namespace",
    "NestedSequence",
    "SupportsBufferProtocol",
]

_T_co = TypeVar("_T_co", covariant=True)

class NestedSequence(Protocol[_T_co]):
    def __getitem__(self, key: int, /) -> _T_co | NestedSequence[_T_co]: ...
    def __len__(self, /) -> int: ...


SupportsBufferProtocol = Any
Array = Any
Device = Any
DType = Any
