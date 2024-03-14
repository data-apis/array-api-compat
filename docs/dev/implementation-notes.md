# Implementation Notes

This page outlines some notes on the implementation of array-api-compat. These
details are not important for users of the package, but they may be useful to
contributors.

## Special Considerations

array-api-compat requires some special development considerations that are
different from most other Python libraries. The goal of array-api-compat is to
be a small library that packages can either vendor or add as a dependency to
implement array API support. Consequently, certain design considerations
should be taken into account:

- *No Hard Dependencies.* Although array-api-compat "depends" on NumPy, CuPy,
  PyTorch, etc., it does not hard depend on them. These libraries are not
  imported unless either an array object is passed to
  {func}`~.array_namespace()`, or the specific `array_api_compat.<namespace>`
  sub-namespace is explicitly imported.

- *Vendorability.* array-api-compat should be [vendorable](vendoring). This
  means that, for instance, all imports in the library are relative imports.
  No code in the package specifically references the name `array_api_compat`
  (we also support renaming the package to something else).
  Vendorability support is tested in `tests/test_vendoring.py`.

- *Pure Python.* To make array-api-compat as easy as possible to add as a
  dependency, the code is all pure Python.

- *Minimal Wrapping Only.* The wrapping functionality is minimal. This means
  that if something is difficult to wrap using pure Python, or if trying to
  support some array API behavior would require a significant amount of code,
  we prefer to leave the behavior as an upstream issue for the array library,
  and [document it as a known difference](../supported-array-libraries.md).

  This also means that we do not at this point in time implement anything
  other than wrappers for functions in the standard, and basic [helper
  functions](../helper-functions.rst) that would be useful for most users of
  array-api-compat. The addition of functions that are not part of the array
  API standard is currently out-of-scope for this package (see the
  [Scope](scope) section of the documentation).

- *No Side-Effects*. array-api-compat behavior should be localized to only the
  specific code that imports and uses it. It should be invisible to end-users
  or users of dependent codes. This in particular implies to the next two
  points.

- *No Monkey Patching.* `array-api-compat` should not attempt to modify
  anything about the underlying library. It is a *wrapper* library only.

- *No Modifying the Array Object.* The array (or tensor) object of the array
  library cannot be modified. This also precludes the creation of array
  subclasses or wrapper classes.

  Any non-standard behavior that is built-in to the array object, such as the
  behavior of [array
  methods](https://data-apis.org/array-api/latest/API_specification/array_object.html),
  is therefore left unwrapped. Users can workaround issues by using
  corresponding [elementwise
  functions](https://data-apis.org/array-api/latest/API_specification/elementwise_functions.html)
  instead of
  [operators](https://data-apis.org/array-api/latest/API_specification/array_object.html#operators),
  and by using the [helper functions](../helper-functions.rst) provided by
  array-api-compat instead of attributes or methods like `x.to_device()`.

- *Avoid Restricting Behavior that is Outside the Scope of the Standard.* All
  array libraries have functions and behaviors that are outside of the scope
  of what is specified by the standard. These behaviors should be left intact
  whenever possible, unless the standard explicitly disallows something. This
  means

  - All namespaces are *extended* with wrapper functions. You may notice the
    extensive use of `import *` in various files in `array_api_compat`. While
    this would normally be questionable, this is the [one actual legitimate
    use-case for `import *`](https://peps.python.org/pep-0008/#imports), to
    re-export names from an external namespace.

  - All wrapper functions pass `**kwargs` through to the wrapped function.

  - Input types not supported by the standard should work if they work in the
    underlying wrapped function (for instance, Python scalars or `np.ndarray`
    subclasses).

  By keeping underlying behaviors intact, it is easier for libraries to swap
  out NumPy or other array libraries for array-api-compat, and it is easier
  for libraries to write array library-specific code paths.

  The onus is on users of array-api-compat to ensure their array API code is
  portable, e.g., by testing against [array-api-strict](array-api-strict).


## Avoiding Code Duplication

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

## Tests

The majority of the behavior for array-api-compat is tested by the
[array-api-tests](https://github.com/data-apis/array-api-tests) test suite for
the array API standard. There are also array-api-compat specific tests in
[`tests/`](https://github.com/data-apis/array-api-compat/tree/main/tests).
These tests should be limited to things that are not tested by the test suite,
e.g., tests for [helper functions](../helper-functions.rst) or for behavior
that is not strictly required by the standard.

array-api-tests is run against all supported libraries are tested on CI
([except for JAX](jax-support)). This is achieved by a [reusable GitHub Actions
Workflow](https://github.com/data-apis/array-api-compat/blob/main/.github/workflows/array-api-tests.yml).
Most libraries have tests that must be xfailed or skipped for various reasons.
These are defined in specific `<library>-xfails.txt` files and are
automatically forwarded to array-api-tests.

You may often need to update these xfail files, either to add new xfails
(e.g., because of new test suite features, or because a test that was
previously thought to be passing actually flaky fails). Try to keep the xfails
files organized, with comments pointing to upstream issues whenever possible.

From time to time, xpass tests should be removed from the xfail files, but be
aware that many xfail tests are flaky, so an xpass should only be removed if
you know that the underlying issue has been fixed.

Array libraries that require a GPU to run (currently only CuPy) cannot be
tested on CI. There is a helper script `test_cupy.sh` that can be used to
manually test CuPy on a machine with a CUDA GPU.
