# Implementation Notes

Since NumPy, CuPy, and to a degree, Dask, are nearly identical in behavior,
most wrapping logic can be shared between them. Wrapped functions that have
the same logic between multiple libraries are in `array_api_compat/common/`.
These functions are defined like

```py
# In array_api_compat/common/_aliases.py

def acos(x, /, xp):
    return xp.arccos(x)
```

The `xp` argument refers to the original array namespace (e.g., `numpy` or
`cupy`). Then in the specific `array_api_compat/numpy/` and
`array_api_compat/cupy/` namespaces, the `@get_xp` decorator is applied to
these functions, which automatically removes the `xp` argument from the
function signature and replaces it with the corresponding array library, like

```py
# In array_api_compat/numpy/_aliases.py

from ..common import _aliases

import numpy as np

acos = get_xp(np)(_aliases.acos)
```

This `acos` now has the signature `acos(x, /)` and calls `numpy.arccos`.

Similarly, for CuPy:

```py
# In array_api_compat/cupy/_aliases.py

from ..common import _aliases

import cupy as cp

acos = get_xp(cp)(_aliases.acos)
```

Most NumPy and CuPy are defined in this way, since their behaviors are nearly
identical PyTorch uses a similar layout in `array_api_compat/torch/`, but it
differs enough from NumPy/CuPy that very few common wrappers for those
libraries are reused. Dask is close to NumPy in behavior and so most Dask
functions also reuse the NumPy/CuPy common wrappers.

Occasionally, a wrapper implementation will need to reference another wrapper
implementation, rather than the base `xp` version. The easiest way to do this
is to call `array_namespace`, like

```py
wrapped_xp = array_namespace(x)
wrapped_xp.wrapped_func(...)
```

Also, if there is a very minor difference required for wrapping, say, CuPy and
NumPy, they can still use a common implementation in `common/_aliases.py` and
use the `is_*_namespace()` or `is_*_function()` [helper
functions](../helper-functions.rst) to branch as necessary.
