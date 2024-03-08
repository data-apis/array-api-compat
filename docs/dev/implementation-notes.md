# Implementation Notes

As noted before, the goal of this library is to reuse the NumPy and CuPy array
objects, rather than wrapping or extending them. This means that the functions
need to accept and return `np.ndarray` for NumPy and `cp.ndarray` for CuPy.

Each namespace (`array_api_compat.numpy`, `array_api_compat.cupy`, and
`array_api_compat.torch`) is populated with the normal library namespace (like
`from numpy import *`). Then specific functions are replaced with wrapped
variants.

Since NumPy and CuPy are nearly identical in behavior, most wrapping logic can
be shared between them. Wrapped functions that have the same logic between
NumPy and CuPy are in `array_api_compat/common/`.
These functions are defined like

```py
# In array_api_compat/common/_aliases.py

def acos(x, /, xp):
    return xp.arccos(x)
```

The `xp` argument refers to the original array namespace (either `numpy` or
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

Since NumPy and CuPy are nearly identical in their behaviors, this allows
writing the wrapping logic for both libraries only once.

PyTorch uses a similar layout in `array_api_compat/torch/`, but it differs
enough from NumPy/CuPy that very few common wrappers for those libraries are
reused.

See https://numpy.org/doc/stable/reference/array_api.html for a full list of
changes from the base NumPy (the differences for CuPy are nearly identical). A
corresponding document does not yet exist for PyTorch, but you can examine the
various comments in the
[implementation](https://github.com/data-apis/array-api-compat/blob/main/array_api_compat/torch/_aliases.py)
to see what functions and behaviors have been wrapped.
