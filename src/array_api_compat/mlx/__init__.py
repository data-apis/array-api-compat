from .._internal import clone_module

__all__ = clone_module("mlx.core", globals())

from . import _aliases
from ._aliases import *  # noqa: F403

# Unsure if this is needed, but it seems to be in the other backends
# __array_api_version__: Final = "2025.12"

__all__ = sorted(set(__all__) | set(_aliases.__all__))


def __dir__() -> list[str]:
    return __all__
