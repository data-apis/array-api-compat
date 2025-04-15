from __future__ import annotations

from types import ModuleType as Namespace
from typing import Any, Protocol, TypeAlias, TypedDict, TypeVar

_T_co = TypeVar("_T_co", covariant=True)


class NestedSequence(Protocol[_T_co]):
    def __getitem__(self, key: int, /) -> _T_co | NestedSequence[_T_co]: ...
    def __len__(self, /) -> int: ...


class SupportsArrayNamespace(Protocol[_T_co]):
    def __array_namespace__(self, /, *, api_version: str | None) -> _T_co: ...


class HasShape(Protocol[_T_co]):
    @property
    def shape(self, /) -> _T_co: ...


Capabilities = TypedDict(
    "Capabilities",
    {
        "boolean indexing": bool,
        "data-dependent shapes": bool,
        "max dimensions": int,
    },
)


SupportsBufferProtocol: TypeAlias = Any
Array: TypeAlias = Any
Device: TypeAlias = Any
DType: TypeAlias = Any


__all__ = [
    "Array",
    "Capabilities",
    "DType",
    "Device",
    "HasShape",
    "Namespace",
    "NestedSequence",
    "SupportsArrayNamespace",
    "SupportsBufferProtocol",
]


def __dir__() -> list[str]:
    return __all__
