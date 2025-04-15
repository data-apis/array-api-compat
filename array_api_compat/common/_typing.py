from __future__ import annotations

from types import ModuleType as Namespace
from typing import Any, Protocol, TypeAlias, TypeVar

__all__ = [
    "Array",
    "SupportsArrayNamespace",
    "DType",
    "Device",
    "HasShape",
    "Namespace",
    "NestedSequence",
    "SupportsBufferProtocol",
]

_T_co = TypeVar("_T_co", covariant=True)

class NestedSequence(Protocol[_T_co]):
    def __getitem__(self, key: int, /) -> _T_co | NestedSequence[_T_co]: ...
    def __len__(self, /) -> int: ...


class SupportsArrayNamespace(Protocol[_T_co]):
    def __array_namespace__(self, /, *, api_version: str | None) -> _T_co: ...


class HasShape(Protocol[_T_co]):
    @property
    def shape(self, /) -> _T_co: ...


SupportsBufferProtocol: TypeAlias = Any
Array: TypeAlias = Any
Device: TypeAlias = Any
DType: TypeAlias = Any
