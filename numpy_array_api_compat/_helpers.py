"""
Various helper functions which are not part of the spec.
"""

from __future__ import annotations

import numpy as np

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
    if isinstance(x, np.ndarray):
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
    if isinstance(x, np.ndarray):
        if stream is not None:
            raise ValueError("The stream argument to to_device() is not supported")
        if device == 'cpu':
            return x
        raise ValueError(f"Unsupported device {device!r}")

    return x.to_device(device, stream=stream)

__all__ = ['device', 'to_device']
