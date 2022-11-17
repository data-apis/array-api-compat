"""
Various helper functions which are not part of the spec.
"""

from __future__ import annotations

import importlib
compat_namespace = importlib.import_module(__package__)

import numpy as np

def _is_numpy_array(x):
    # TODO: Should we reject ndarray subclasses?
    return isinstance(x, (np.ndarray, np.generic))

def is_array_api_obj(x):
    """
    Check if x is an array API compatible array object.
    """
    return _is_numpy_array(x) or hasattr(x, '__array_namespace__')

def get_namespace(*xs, _use_compat=True):
    """
    Get the array API compatible namespace for the arrays `xs`.

    `xs` should contain one or more arrays.
    """
    namespaces = set()
    for x in xs:
        if isinstance(x, (tuple, list)):
            namespaces.add(get_namespace(*x, _use_compat=_use_compat))
        elif hasattr(x, '__array_namespace__'):
            namespaces.add(x.__array_namespace__)
        elif _is_numpy_array(x):
            if _use_compat:
                namespaces.add(compat_namespace)
            else:
                namespaces.add(np)
        else:
            # TODO: Support Python scalars?
            raise ValueError("The input is not a supported array type")

    if not namespaces:
        raise ValueError("Unrecognized array input")

    if len(namespaces) != 1:
        raise ValueError(f"Multiple namespaces for array inputs: {namespaces}")

    xp, = namespaces

    return xp

# device and to_device are not included in array object of this library
# because this library just reuses ndarray without wrapping or subclassing it.
# These helper functions can be used instead of the wrapper functions for
# libraries that need to support both NumPy and other libraries that use devices.
def device(x: "Array", /) -> "Device":
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
    if _is_numpy_array(x):
        return "cpu"
    return x.device

def to_device(x: "Array", device: "Device", /, *, stream: Optional[Union[int, Any]] = None) -> "Array":
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
    if _is_numpy_array(x):
        if stream is not None:
            raise ValueError("The stream argument to to_device() is not supported")
        if device == 'cpu':
            return x
        raise ValueError(f"Unsupported device {device!r}")

    return x.to_device(device, stream=stream)

__all__ = ['is_array_api_obj', 'get_namespace', 'device', 'to_device']
