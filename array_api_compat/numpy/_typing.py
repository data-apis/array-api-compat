from __future__ import annotations

__all__ = ["Array", "DType", "Device"]
_all_ignore = ["np"]

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

Device = Literal["cpu"]
if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    # NumPy 1.x on Python 3.10 fails to parse np.dtype[]
    DType: TypeAlias = np.dtype[
        np.bool_
        | np.integer[Any]
        | np.float32
        | np.float64
        | np.complex64
        | np.complex128
    ]
    Array: TypeAlias = np.ndarray[Any, DType]
else:
    DType = np.dtype
    Array = np.ndarray
