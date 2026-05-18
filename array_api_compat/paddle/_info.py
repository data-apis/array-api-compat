"""
Array API Inspection namespace

This is the namespace for inspection functions as defined by the array API
standard. See
https://data-apis.org/array-api/latest/API_specification/inspection.html for
more details.

"""

import paddle

from functools import cache


class __array_namespace_info__:
    """
    Get the array API inspection namespace for Paddle.

    The array API inspection namespace defines the following functions:

    - capabilities()
    - default_device()
    - default_dtypes()
    - dtypes()
    - devices()

    See
    https://data-apis.org/array-api/latest/API_specification/inspection.html
    for more details.

    Returns
    -------
    info : ModuleType
        The array API inspection namespace for Paddle.

    Examples
    --------
    >>> info = np.__array_namespace_info__()
    >>> info.default_dtypes()
    {'real floating': numpy.float64,
     'complex floating': numpy.complex128,
     'integral': numpy.int64,
     'indexing': numpy.int64}

    """

    __module__ = "paddle"

    def capabilities(self):
        """
        Return a dictionary of array API library capabilities.

        The resulting dictionary has the following keys:

        - **"boolean indexing"**: boolean indicating whether an array library
          supports boolean indexing. Always ``True`` for Paddle.

        - **"data-dependent shapes"**: boolean indicating whether an array
          library supports data-dependent output shapes. Always ``True`` for
          Paddle.

        See
        https://data-apis.org/array-api/latest/API_specification/generated/array_api.info.capabilities.html
        for more details.

        See Also
        --------
        __array_namespace_info__.default_device,
        __array_namespace_info__.default_dtypes,
        __array_namespace_info__.dtypes,
        __array_namespace_info__.devices

        Returns
        -------
        capabilities : dict
            A dictionary of array API library capabilities.

        Examples
        --------
        >>> info = np.__array_namespace_info__()
        >>> info.capabilities()
        {'boolean indexing': True,
         'data-dependent shapes': True}

        """
        return {
            "boolean indexing": True,
            "data-dependent shapes": True,
            # 'max rank' will be part of the 2024.12 standard
            # "max rank": 64,
        }

    def default_device(self):
        """
        The default device used for new Paddle arrays.

        See Also
        --------
        __array_namespace_info__.capabilities,
        __array_namespace_info__.default_dtypes,
        __array_namespace_info__.dtypes,
        __array_namespace_info__.devices

        Returns
        -------
        device : str
            The default device used for new Paddle arrays.

        Examples
        --------
        >>> info = np.__array_namespace_info__()
        >>> info.default_device()
        'cpu'

        """
        return paddle.device.get_device()

    def default_dtypes(self, *, device=None):
        """
        The default data types used for new Paddle arrays.

        Parameters
        ----------
        device : str, optional
            The device to get the default data types for. For Paddle, only
            ``'cpu'`` is allowed.

        Returns
        -------
        dtypes : dict
            A dictionary describing the default data types used for new Paddle
            arrays.

        See Also
        --------
        __array_namespace_info__.capabilities,
        __array_namespace_info__.default_device,
        __array_namespace_info__.dtypes,
        __array_namespace_info__.devices

        Examples
        --------
        >>> info = np.__array_namespace_info__()
        >>> info.default_dtypes()
        {'real floating': paddle.float32,
         'complex floating': paddle.complex64,
         'integral': paddle.int64,
         'indexing': paddle.int64}

        """
        # Note: if the default is set to float64, the devices like MPS that
        # don't support float64 will error. We still return the default_dtype
        # value here because this error doesn't represent a different default
        # per-device.
        default_floating = paddle.get_default_dtype()
        if default_floating in ["float16", "float32", "float64", "bfloat16"]:
            default_floating = getattr(paddle, default_floating)
        else:
            raise ValueError(f"Unsupported default floating: {default_floating}")
        default_complex = (
            paddle.complex64
            if default_floating == paddle.float32
            else paddle.complex128
        )
        default_integral = paddle.int64
        return {
            "real floating": default_floating,
            "complex floating": default_complex,
            "integral": default_integral,
            "indexing": default_integral,
        }

    def _dtypes(self, kind):
        bool = paddle.bool
        int8 = paddle.int8
        int16 = paddle.int16
        int32 = paddle.int32
        int64 = paddle.int64
        uint8 = paddle.uint8
        # uint16, uint32, and uint64 are not fully supported in paddle,
        # we omit them from this function.
        float32 = paddle.float32
        float64 = paddle.float64
        complex64 = paddle.complex64
        complex128 = paddle.complex128

        if kind is None:
            return {
                "bool": bool,
                "int8": int8,
                "int16": int16,
                "int32": int32,
                "int64": int64,
                "uint8": uint8,
                "float32": float32,
                "float64": float64,
                "complex64": complex64,
                "complex128": complex128,
            }
        if kind == "bool":
            return {"bool": bool}
        if kind == "signed integer":
            return {
                "int8": int8,
                "int16": int16,
                "int32": int32,
                "int64": int64,
            }
        if kind == "unsigned integer":
            return {
                "uint8": uint8,
            }
        if kind == "integral":
            return {
                "int8": int8,
                "int16": int16,
                "int32": int32,
                "int64": int64,
                "uint8": uint8,
            }
        if kind == "real floating":
            return {
                "float32": float32,
                "float64": float64,
            }
        if kind == "complex floating":
            return {
                "complex64": complex64,
                "complex128": complex128,
            }
        if kind == "numeric":
            return {
                "int8": int8,
                "int16": int16,
                "int32": int32,
                "int64": int64,
                "uint8": uint8,
                "float32": float32,
                "float64": float64,
                "complex64": complex64,
                "complex128": complex128,
            }
        if isinstance(kind, tuple):
            res = {}
            for k in kind:
                res.update(self.dtypes(kind=k))
            return res
        raise ValueError(f"unsupported kind: {kind!r}")

    @cache
    def dtypes(self, *, device=None, kind=None):
        """
        The array API data types supported by Paddle.

        Note that this function only returns data types that are defined by
        the array API.

        Parameters
        ----------
        device : str, optional
            The device to get the data types for.
        kind : str or tuple of str, optional
            The kind of data types to return. If ``None``, all data types are
            returned. If a string, only data types of that kind are returned.
            If a tuple, a dictionary containing the union of the given kinds
            is returned. The following kinds are supported:

            - ``'bool'``: boolean data types (i.e., ``bool``).
            - ``'signed integer'``: signed integer data types (i.e., ``int8``,
              ``int16``, ``int32``, ``int64``).
            - ``'unsigned integer'``: unsigned integer data types (i.e.,
              ``uint8``, ``uint16``, ``uint32``, ``uint64``).
            - ``'integral'``: integer data types. Shorthand for ``('signed
              integer', 'unsigned integer')``.
            - ``'real floating'``: real-valued floating-point data types
              (i.e., ``float32``, ``float64``).
            - ``'complex floating'``: complex floating-point data types (i.e.,
              ``complex64``, ``complex128``).
            - ``'numeric'``: numeric data types. Shorthand for ``('integral',
              'real floating', 'complex floating')``.

        Returns
        -------
        dtypes : dict
            A dictionary mapping the names of data types to the corresponding
            Paddle data types.

        See Also
        --------
        __array_namespace_info__.capabilities,
        __array_namespace_info__.default_device,
        __array_namespace_info__.default_dtypes,
        __array_namespace_info__.devices

        Examples
        --------
        >>> info = np.__array_namespace_info__()
        >>> info.dtypes(kind='signed integer')
        {'int8': numpy.int8,
         'int16': numpy.int16,
         'int32': numpy.int32,
         'int64': numpy.int64}

        """
        res = self._dtypes(kind)
        for k, v in res.copy().items():
            try:
                paddle.empty((0,), dtype=v, device=device)
            except:
                del res[k]
        return res

    @cache
    def devices(self):
        """
        The devices supported by Paddle.

        Returns
        -------
        devices : list of str
            The devices supported by Paddle.

        See Also
        --------
        __array_namespace_info__.capabilities,
        __array_namespace_info__.default_device,
        __array_namespace_info__.default_dtypes,
        __array_namespace_info__.dtypes

        Examples
        --------
        >>> info = np.__array_namespace_info__()
        >>> info.devices()
        [device(type='cpu'), device(type='mps', index=0), device(type='meta')]

        """
        # Paddle doesn't have a straightforward way to get the list of all
        # currently supported devices. To do this, we first parse the error
        # message of paddle.device to get the list of all possible types of
        # device:
        try:
            paddle.set_device("notadevice")
        except ValueError as e:
            # The error message is something like:
            # ValueError: The device must be a string which is like 'cpu', 'gpu', 'gpu:x', 'xpu', 'xpu:x', 'npu', 'npu:x
            devices_names = (
                e.args[0]
                .split("The device must be a string which is like ")[1]
                .split(", ")
            )
            devices_names = [
                name.strip("'") for name in devices_names if ":" not in name
            ]

        # Next we need to check for different indices for different devices.
        # device(device_name, index=index) doesn't actually check if the
        # device name or index is valid. We have to try to create a tensor
        # with it (which is why this function is cached).
        devices = []
        for device_name in devices_names:
            i = 0
            while True:
                try:
                    if device_name == "cpu":
                        a = paddle.empty((0,), place=paddle.CPUPlace())
                    elif device_name == "gpu":
                        a = paddle.empty((0,), place=paddle.CUDAPlace(i))
                    elif device_name == "xpu":
                        a = paddle.empty((0,), place=paddle.XPUPlace())
                    else:
                        raise
                    if a.place in devices:
                        break
                    devices.append(a.device)
                except:
                    break
                i += 1

        return devices
