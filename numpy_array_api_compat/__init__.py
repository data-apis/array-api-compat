"""
NumPy Array API compatibility library

This is a small wrapper around NumPy that is compatible with the Array API
standard https://data-apis.org/array-api/latest/. See also NEP 47
https://numpy.org/neps/nep-0047-array-api-standard.html.

Unlike numpy.array_api, this is not a strict minimal implementation of the
Array API, but rather just an extension of the main NumPy namespace with
changes needed to be compliant with the Array API. See
https://numpy.org/doc/stable/reference/array_api.html for a full list of
changes. In particular, unlike numpy.array_api, this package does not use a
separate Array object, but rather just uses numpy.ndarray directly.

Library authors using the Array API may wish to test against numpy.array_api
to ensure they are not using functionality outside of the standard, but prefer
this implementation for the default when working with NumPy arrays.

In addition, several helper functions are provided in this library which are
not part of the array API specification but which are useful for libraries
writing against the array API specification who wish to support NumPy and
other array API compatible libraries.

Known differences from the Array API spec:

- The array methods __array_namespace__, device, to_device, and mT are not
  defined. This reuses np.ndarray and we don't want to monkeypatch or wrap it.
  The helper functions device() and to_device() are provided to work around
  these missing methods. x.mT can be replaced with
  xp.linalg.matrix_transpose(x).

- NumPy value-based casting for scalars will be in effect unless explicitly
  disabled with the environment variable NPY_PROMOTION_STATE=weak or
  np._set_promotion_state('weak') (requires NumPy 1.24 or newer, see NEP 50
  and https://github.com/numpy/numpy/issues/22341)

- NumPy functions which are not wrapped may not have the same type annotations
  as the spec.

- NumPy functions which are not wrapped may not use positional-only arguments.

"""

from numpy import *

# These imports may overwrite names from the import * above.
from ._aliases import *

# Don't know why, but we have to do this to import linalg. If we instead do
#
# from . import linalg
#
# It doesn't overwrite np.linalg from above.
import numpy_array_api_compat.linalg

from .linalg import matrix_transpose, vecdot

from .helpers import *
