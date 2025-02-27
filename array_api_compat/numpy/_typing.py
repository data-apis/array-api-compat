from __future__ import annotations

__all__ = ["Array", "DType", "Device"]
_all_ignore = ["np"]

from typing import Literal, Union

import numpy as np
from numpy import ndarray as Array

Device = Literal["cpu"]
try:
    DType = np.dtype[
        Union[
            np.intp,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.float32,
            np.float64,
            np.complex64,
            np.complex128,
            np.bool_,
        ]
    ]
except TypeError:
    # NumPy 1.x on Python 3.9 and 3.10
    DType = np.dtype
