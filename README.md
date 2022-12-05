# Array API compatibility library

This is a small wrapper around NumPy and CuPy that is compatible with the
[Array API standard](https://data-apis.org/array-api/latest/). See also [NEP
47](https://numpy.org/neps/nep-0047-array-api-standard.html).

Unlike `numpy.array_api`, this is not a strict minimal implementation of the
Array API, but rather just an extension of the main NumPy and CuPy namespaces
with changes needed to be compliant with the Array API.

Library authors using the Array API may wish to test against `numpy.array_api`
to ensure they are not using functionality outside of the standard, but prefer
this implementation for the default when working with NumPy or CuPy arrays.

See https://numpy.org/doc/stable/reference/array_api.html for a full list of
changes. In particular, unlike `numpy.array_api`, this package does not use a
separate Array object, but rather just uses `numpy.ndarray` directly.

Note that some of the functionality in this library is backwards incompatible
with NumPy.

This library also supports CuPy in addition to NumPy. If you want support for
other array libraries, please [open an
issue](https://github.com/data-apis/array-api-compat/issues).

Library authors using the Array API may wish to test against `numpy.array_api`
to ensure they are not using functionality outside of the standard, but prefer
this implementation for end users who use NumPy arrays.

## Usage

To use this library replace

```py
import numpy as np
```

with

```py
import array_api_compat.numpy as np
```

and replace

```py
import cupy as cp
```

with

```py
import array_api_compat.cupy as cp
```

Each will include all the functions from the normal NumPy/CuPy namespace,
except that functions that are part of the array API are wrapped so that they
have the correct array API behavior. In each case, the array object used will
be thew same array object from the wrapped library.


## Helper Functions

In addition to the default NumPy/CuPy namespace and functions in the array API
specification, there are several helper functions
included that aren't part of the specification but which are useful for using
the array API:

- `is_array_api_obj(x)`: Return `True` if `x` is an array API compatible array
  object.

- `get_namespace(*xs)`: Get the corresponding array API namespace for the
  arrays `xs`. If the arrays are NumPy or CuPy arrays, the returned namespace
  will be `array_api_compat.numpy` or `array_api_compat.cupy` so that it is
  array API compatible.

- `device(x)`: Equivalent to
  [`x.device`](https://data-apis.org/array-api/latest/API_specification/generated/signatures.array_object.array.device.html)
  in the array API specification. Included because `numpy.ndarray` does not
  include the `device` attribute and this library does not wrap or extend the
  array object. Note that for NumPy, `device` is always `"cpu"`.

- `to_device(x, device, /, *, stream=None)`: Equivalent to
  [`x.to_device`](https://data-apis.org/array-api/latest/API_specification/generated/signatures.array_object.array.to_device.html).
  Included because neither NumPy's nor CuPy's ndarray objects include this
  method. For NumPy, this function effectively does nothing since the only
  supported device is the CPU, but for CuPy, this method supports CuPy CUDA
  [Device](https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.Device.html)
  and
  [Stream](https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.Stream.html)
  objects.

## Known Differences from the Array API Specification

There are some known differences between this library and the array API
specification:

- The array methods `__array_namespace__`, `device` (for NumPy), `to_device`,
  and `mT` are not defined. This reuses `np.ndarray` and `cp.ndarray` and we
  don't want to monkeypatch or wrap it. The helper functions `device()` and
  `to_device()` are provided to work around these missing methods (see above).
  `x.mT` can be replaced with `xp.linalg.matrix_transpose(x)`.
  `get_namespace(x)` should be used instead of `x.__array_namespace__`.

- NumPy value-based casting for scalars will be in effect unless explicitly
  disabled with the environment variable NPY_PROMOTION_STATE=weak or
  np._set_promotion_state('weak') (requires NumPy 1.24 or newer, see NEP 50
  and https://github.com/numpy/numpy/issues/22341)

- Functions which are not wrapped may not have the same type annotations
  as the spec.

- Functions which are not wrapped may not use positional-only arguments.

## Vendoring

This library supports vendoring as an installation method. To vendor the
library, simply copy `array_api_compat` into the appropriate place in the
library, like

```
cp -R array_api_compat/ mylib/vendored/array_api_compat
```

You may also rename it to something else if you like (nowhere in the code
references the name "array_api_compat").

Alternatively, the library may be installed as dependency on PyPI.

## Implementation

As noted before, the goal of this library is to reuse the NumPy and CuPy array
objects, rather than wrapping or extending them. This means that the functions
need to accept and return `np.ndarray` for NumPy and `cp.ndarray` for CuPy.

Each namespace (`array_api_compat.numpy` and `array_api_compat.cupy`) is
populated with the normal library namespace (like `from numpy import *`). Then
specific functions are replaced with wrapped variants. Wrapped functions that
have the same logic between NumPy and CuPy (which is most functions) are in
`array_api_compat/common/`. These functions are defined like

```py
# In array_api_compat/common/_aliases.py

def acos(x, /, xp):
    return xp.arccos(x)
```

The `xp` argument refers to the original array namespace (either `numpy` or
`cupy`). Then in the specific `array_api_compat/numpy` and
`array_api_compat/cupy` namespace, the `get_xp` decorator is applied to these
functions, which automatically removes the `xp` argument from the function
signature and replaces it with the corresponding array library, like

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
writing the wrapping logic for both libraries only once. If support is added
for other libraries which differ significantly from NumPy, their wrapper code
should go in their specific sub-namespace instead of `common/`.
