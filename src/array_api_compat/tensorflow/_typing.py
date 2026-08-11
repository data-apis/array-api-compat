__all__ = ["Array", "Device", "DType"]

from typing import TypeAlias

from tensorflow import DType, Tensor

Array = Tensor
Device: TypeAlias = str
