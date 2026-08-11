from collections.abc import Sequence
from typing import Final

import tensorflow as _tf

from .._internal import clone_module

__all__ = clone_module("tensorflow", globals())

# TensorShape is not the array object; this lets shape helpers consume x.shape
# without mutating TensorFlow tensors.
Sequence.register(type(_tf.TensorShape(())))

# These imports may overwrite names from the import * above.
from . import _aliases
from ._aliases import *  # noqa: F403
from ._info import __array_namespace_info__  # noqa: F401

__import__(__spec__.parent + ".linalg")
__import__(__spec__.parent + ".fft")

__array_api_version__: Final = "2025.12"

__all__ = sorted(
    set(__all__)
    | set(_aliases.__all__)
    | {"__array_api_version__", "__array_namespace_info__", "linalg", "fft"}
)


def __dir__() -> list[str]:
    return __all__
