"""
Array API Inspection namespace for MLX.

See https://data-apis.org/array-api/latest/API_specification/inspection.html
for more details.
"""
from __future__ import annotations

import mlx.core as mx

from ..common._typing import DefaultDTypes
from ._typing import Device, DType


class __array_namespace_info__:
    """
    Get the array API inspection namespace for MLX.

    The array API inspection namespace defines the following functions:

    - capabilities()
    - default_device()
    - default_dtypes()
    - dtypes()
    - devices()

    See
    https://data-apis.org/array-api/latest/API_specification/inspection.html
    for more details.
    """

    __module__ = "mlx.core"

    def capabilities(self) -> dict:
        """
        Return a dictionary of array API library capabilities.

        For MLX:
        - "boolean indexing": True
        - "data-dependent shapes": False (MLX uses lazy evaluation with static shapes)
        - "max dimensions": 16 (MLX practical limit)

        See
        https://data-apis.org/array-api/latest/API_specification/generated/array_api.info.capabilities.html
        for more details.
        """
        return {
            "boolean indexing": True,
            # MLX uses lazy/static shapes; nonzero() output size is
            # not known at trace time, so data-dependent shapes are not supported.
            "data-dependent shapes": False,
            "max dimensions": 16,
        }

    def default_device(self) -> Device:
        """
        The default device used for new MLX arrays.

        Returns the MLX default device (typically the GPU on Apple Silicon).
        """
        return mx.default_device()

    def default_dtypes(
        self,
        *,
        device: Device | None = None,
    ) -> DefaultDTypes:
        """
        The default data types used for new MLX arrays.

        MLX defaults:
        - "real floating": float32 (MLX has no float64 by default)
        - "complex floating": complex64
        - "integral": int32
        - "indexing": int32
        """
        # MLX doesn't validate devices per call; unified memory model.
        return {
            "real floating": mx.float32,
            "complex floating": mx.complex64,
            "integral": mx.int32,
            "indexing": mx.int32,
        }

    def dtypes(
        self,
        *,
        device: Device | None = None,
        kind: str | tuple[str, ...] | None = None,
    ) -> dict[str, DType]:
        """
        The array API data types supported by MLX.

        Note: MLX does not support float64 or complex128.

        Parameters
        ----------
        device : optional
            Ignored (MLX uses unified memory).
        kind : str or tuple of str, optional
            Filter by kind: 'bool', 'signed integer', 'unsigned integer',
            'integral', 'real floating', 'complex floating', 'numeric'.
        """
        _bool = {"bool": mx.bool_}
        _signed = {
            "int8": mx.int8,
            "int16": mx.int16,
            "int32": mx.int32,
            "int64": mx.int64,
        }
        _unsigned = {
            "uint8": mx.uint8,
            "uint16": mx.uint16,
            "uint32": mx.uint32,
            "uint64": mx.uint64,
        }
        # MLX has no float64/complex128
        _real_float = {
            "float16": mx.float16,
            "bfloat16": mx.bfloat16,
            "float32": mx.float32,
        }
        _complex_float = {
            "complex64": mx.complex64,
        }

        if kind is None:
            return {**_bool, **_signed, **_unsigned, **_real_float, **_complex_float}
        if kind == "bool":
            return _bool
        if kind == "signed integer":
            return _signed
        if kind == "unsigned integer":
            return _unsigned
        if kind == "integral":
            return {**_signed, **_unsigned}
        if kind == "real floating":
            return _real_float
        if kind == "complex floating":
            return _complex_float
        if kind == "numeric":
            return {**_signed, **_unsigned, **_real_float, **_complex_float}
        if isinstance(kind, tuple):
            res: dict[str, DType] = {}
            for k in kind:
                res.update(self.dtypes(kind=k))
            return res
        raise ValueError(f"unsupported kind: {kind!r}")

    def devices(self) -> list[Device]:
        """
        The devices supported by MLX.

        Returns CPU and GPU devices available on the current machine.
        """
        # Return both available MLX device types
        return [mx.Device(mx.DeviceType.cpu), mx.Device(mx.DeviceType.gpu)]


__all__ = ["__array_namespace_info__"]


def __dir__() -> list[str]:
    return __all__
