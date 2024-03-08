"""
Various helper functions which are not part of the spec.

Functions which start with an underscore are for internal use only but helpers
that are in __all__ are intended as additional helper functions for use by end
users of the compat library.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional, Union, Any
    from ._typing import Array, Device

import sys
import math
import inspect
import warnings

def is_numpy_array(x):
    # Avoid importing NumPy if it isn't already
    if 'numpy' not in sys.modules:
        return False

    import numpy as np

    # TODO: Should we reject ndarray subclasses?
    return isinstance(x, (np.ndarray, np.generic))

def is_cupy_array(x):
    # Avoid importing NumPy if it isn't already
    if 'cupy' not in sys.modules:
        return False

    import cupy as cp

    # TODO: Should we reject ndarray subclasses?
    return isinstance(x, (cp.ndarray, cp.generic))

def is_torch_array(x):
    # Avoid importing torch if it isn't already
    if 'torch' not in sys.modules:
        return False

    import torch

    # TODO: Should we reject ndarray subclasses?
    return isinstance(x, torch.Tensor)

def is_dask_array(x):
    # Avoid importing dask if it isn't already
    if 'dask.array' not in sys.modules:
        return False

    import dask.array

    return isinstance(x, dask.array.Array)

def is_jax_array(x):
    # Avoid importing jax if it isn't already
    if 'jax' not in sys.modules:
        return False

    import jax

    return isinstance(x, jax.Array)

def is_array_api_obj(x):
    """
    Check if x is an array API compatible array object.
    """
    return is_numpy_array(x) \
        or is_cupy_array(x) \
        or is_torch_array(x) \
        or is_dask_array(x) \
        or is_jax_array(x) \
        or hasattr(x, '__array_namespace__')

def _check_api_version(api_version):
    if api_version == '2021.12':
        warnings.warn("The 2021.12 version of the array API specification was requested but the returned namespace is actually version 2022.12")
    elif api_version is not None and api_version != '2022.12':
        raise ValueError("Only the 2022.12 version of the array API specification is currently supported")

def array_namespace(*xs, api_version=None, _use_compat=True):
    """
    Get the array API compatible namespace for the arrays `xs`.

    `xs` should contain one or more arrays.

    Typical usage is

        def your_function(x, y):
            xp = array_api_compat.array_namespace(x, y)
            # Now use xp as the array library namespace
            return xp.mean(x, axis=0) + 2*xp.std(y, axis=0)

    api_version should be the newest version of the spec that you need support
    for (currently the compat library wrapped APIs only support v2021.12).
    """
    namespaces = set()
    for x in xs:
        if is_numpy_array(x):
            _check_api_version(api_version)
            if _use_compat:
                from .. import numpy as numpy_namespace
                namespaces.add(numpy_namespace)
            else:
                import numpy as np
                namespaces.add(np)
        elif is_cupy_array(x):
            _check_api_version(api_version)
            if _use_compat:
                from .. import cupy as cupy_namespace
                namespaces.add(cupy_namespace)
            else:
                import cupy as cp
                namespaces.add(cp)
        elif is_torch_array(x):
            _check_api_version(api_version)
            if _use_compat:
                from .. import torch as torch_namespace
                namespaces.add(torch_namespace)
            else:
                import torch
                namespaces.add(torch)
        elif is_dask_array(x):
            _check_api_version(api_version)
            if _use_compat:
                from ..dask import array as dask_namespace
                namespaces.add(dask_namespace)
            else:
                raise TypeError("_use_compat cannot be False if input array is a dask array!")
        elif is_jax_array(x):
            _check_api_version(api_version)
            # jax.experimental.array_api is already an array namespace. We do
            # not have a wrapper submodule for it.
            import jax.experimental.array_api as jnp
            namespaces.add(jnp)
        elif hasattr(x, '__array_namespace__'):
            namespaces.add(x.__array_namespace__(api_version=api_version))
        else:
            # TODO: Support Python scalars?
            raise TypeError(f"{type(x).__name__} is not a supported array type")

    if not namespaces:
        raise TypeError("Unrecognized array input")

    if len(namespaces) != 1:
        raise TypeError(f"Multiple namespaces for array inputs: {namespaces}")

    xp, = namespaces

    return xp

# backwards compatibility alias
get_namespace = array_namespace

def _check_device(xp, device):
    if xp == sys.modules.get('numpy'):
        if device not in ["cpu", None]:
            raise ValueError(f"Unsupported device for NumPy: {device!r}")

# Placeholder object to represent the dask device
# when the array backend is not the CPU.
# (since it is not easy to tell which device a dask array is on)
class _dask_device:
    def __repr__(self):
        return "DASK_DEVICE"

_DASK_DEVICE = _dask_device()

# device() is not on numpy.ndarray or dask.array and to_device() is not on numpy.ndarray
# or cupy.ndarray. They are not included in array objects of this library
# because this library just reuses the respective ndarray classes without
# wrapping or subclassing them. These helper functions can be used instead of
# the wrapper functions for libraries that need to support both NumPy/CuPy and
# other libraries that use devices.
def device(x: Array, /) -> Device:
    """
    Hardware device the array data resides on.

    Parameters
    ----------
    x: array
        array instance from NumPy or an array API compatible library.

    Returns
    -------
    out: device
        a ``device`` object (see the "Device Support" section of the array API specification).
    """
    if is_numpy_array(x):
        return "cpu"
    elif is_dask_array(x):
        # Peek at the metadata of the jax array to determine type
        try:
            import numpy as np
            if isinstance(x._meta, np.ndarray):
                # Must be on CPU since backed by numpy
                return "cpu"
        except ImportError:
            pass
        return _DASK_DEVICE
    elif is_jax_array(x):
        # JAX has .device() as a method, but it is being deprecated so that it
        # can become a property, in accordance with the standard. In order for
        # this function to not break when JAX makes the flip, we check for
        # both here.
        if inspect.ismethod(x.device):
            return x.device()
        else:
            return x.device
    return x.device

# Based on cupy.array_api.Array.to_device
def _cupy_to_device(x, device, /, stream=None):
    import cupy as cp
    from cupy.cuda import Device as _Device
    from cupy.cuda import stream as stream_module
    from cupy_backends.cuda.api import runtime

    if device == x.device:
        return x
    elif device == "cpu":
        # allowing us to use `to_device(x, "cpu")`
        # is useful for portable test swapping between
        # host and device backends
        return x.get()
    elif not isinstance(device, _Device):
        raise ValueError(f"Unsupported device {device!r}")
    else:
        # see cupy/cupy#5985 for the reason how we handle device/stream here
        prev_device = runtime.getDevice()
        prev_stream: stream_module.Stream = None
        if stream is not None:
            prev_stream = stream_module.get_current_stream()
            # stream can be an int as specified in __dlpack__, or a CuPy stream
            if isinstance(stream, int):
                stream = cp.cuda.ExternalStream(stream)
            elif isinstance(stream, cp.cuda.Stream):
                pass
            else:
                raise ValueError('the input stream is not recognized')
            stream.use()
        try:
            runtime.setDevice(device.id)
            arr = x.copy()
        finally:
            runtime.setDevice(prev_device)
            if stream is not None:
                prev_stream.use()
        return arr

def _torch_to_device(x, device, /, stream=None):
    if stream is not None:
        raise NotImplementedError
    return x.to(device)

def to_device(x: Array, device: Device, /, *, stream: Optional[Union[int, Any]] = None) -> Array:
    """
    Copy the array from the device on which it currently resides to the specified ``device``.

    Parameters
    ----------
    x: array
        array instance from NumPy or an array API compatible library.
    device: device
        a ``device`` object (see the "Device Support" section of the array API specification).
    stream: Optional[Union[int, Any]]
        stream object to use during copy. In addition to the types supported in ``array.__dlpack__``, implementations may choose to support any library-specific stream object with the caveat that any code using such an object would not be portable.

    Returns
    -------
    out: array
        an array with the same data and data type as ``x`` and located on the specified ``device``.

    .. note::
       If ``stream`` is given, the copy operation should be enqueued on the provided ``stream``; otherwise, the copy operation should be enqueued on the default stream/queue. Whether the copy is performed synchronously or asynchronously is implementation-dependent. Accordingly, if synchronization is required to guarantee data safety, this must be clearly explained in a conforming library's documentation.
    """
    if is_numpy_array(x):
        if stream is not None:
            raise ValueError("The stream argument to to_device() is not supported")
        if device == 'cpu':
            return x
        raise ValueError(f"Unsupported device {device!r}")
    elif is_cupy_array(x):
        # cupy does not yet have to_device
        return _cupy_to_device(x, device, stream=stream)
    elif is_torch_array(x):
        return _torch_to_device(x, device, stream=stream)
    elif is_dask_array(x):
        if stream is not None:
            raise ValueError("The stream argument to to_device() is not supported")
        # TODO: What if our array is on the GPU already?
        if device == 'cpu':
            return x
        raise ValueError(f"Unsupported device {device!r}")
    elif is_jax_array(x):
        # This import adds to_device to x
        import jax.experimental.array_api # noqa: F401
        return x.to_device(device, stream=stream)
    return x.to_device(device, stream=stream)

def size(x):
    """
    Return the total number of elements of x
    """
    if None in x.shape:
        return None
    return math.prod(x.shape)

__all__ = [
    "array_namespace",
    "device",
    "get_namespace",
    "is_array_api_obj",
    "is_cupy_array",
    "is_dask_array",
    "is_jax_array",
    "is_numpy_array",
    "is_torch_array",
    "size",
    "to_device",
]

_all_ignore = ['sys', 'math', 'inspect', 'warnings']
