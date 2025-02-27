# Array API compatibility library

This is a small wrapper around common array libraries that is compatible with
the [Array API standard](https://data-apis.org/array-api/latest/). Currently,
NumPy, CuPy, PyTorch, Dask, JAX, ndonnx, and Sparse are supported. If you want
support for other array libraries, or if you encounter any issues, please
[open an issue](https://github.com/data-apis/array-api-compat/issues).

Note that some of the functionality in this library is backwards incompatible
with the corresponding wrapped libraries. The end-goal is to eventually make
each array library itself fully compatible with the array API, but this
requires making backwards incompatible changes in many cases, so this will
take some time.

Currently all libraries here are implemented against the [2024.12
version](https://data-apis.org/array-api/2024.12/) of the standard.

## Installation

`array-api-compat` is available on both [PyPI](https://pypi.org/project/array-api-compat/)

```
python -m pip install array-api-compat
```

and [conda-forge](https://anaconda.org/conda-forge/array-api-compat)

```
conda install --channel conda-forge array-api-compat
```

## Usage

The typical usage of this library will be to get the corresponding array API
compliant namespace from the input arrays using {func}`~.array_namespace()`, like

```py
def your_function(x, y):
    xp = array_api_compat.array_namespace(x, y)
    # Now use xp as the array library namespace
    return xp.mean(x, axis=0) + 2*xp.std(y, axis=0)
```

If you wish to have library-specific code-paths, you can import the
corresponding wrapped namespace for each library, like

```py
import array_api_compat.numpy as np
```

```py
import array_api_compat.cupy as cp
```

```py
import array_api_compat.torch as torch
```

```py
import array_api_compat.dask as da
```

```{note}
There are no `array_api_compat` submodules for JAX, sparse, or ndonnx. These
support for these libraries is contained in the libraries themselves (JAX
support is in the `jax.numpy` module in JAX v0.4.32 or newer, and in the
`jax.experimental.array_api` module for older JAX versions). The
array-api-compat support for these libraries consists of supporting them in
the [helper functions](helper-functions).
```

Each will include all the functions from the normal NumPy/CuPy/PyTorch/dask.array
namespace, except that functions that are part of the array API are wrapped so
that they have the correct array API behavior. In each case, the array object
used will be the same array object from the wrapped library.

(array-api-strict)=
## Difference between `array_api_compat` and `array_api_strict`

[`array_api_strict`](https://data-apis.org/array-api-strict/) is a
strict minimal implementation of the array API standard, formerly known as
`numpy.array_api` (see [NEP
47](https://numpy.org/neps/nep-0047-array-api-standard.html)). For example,
`array_api_strict` does not include any functions that are not part of the
array API specification, and will explicitly disallow behaviors that are not
required by the spec (e.g., [cross-kind type
promotions](https://data-apis.org/array-api/latest/API_specification/type_promotion.html)).
(`cupy.array_api` is similar to `array_api_strict`)

`array_api_compat`, on the other hand, is just an extension of the
corresponding array library namespaces with changes needed to be compliant
with the array API. It includes all additional library functions not mentioned
in the spec, and allows any library behaviors not explicitly disallowed by it,
such as cross-kind casting.

In particular, unlike `array_api_strict`, this package does not use a separate
`Array` object, but rather just uses the corresponding array library array
objects (`numpy.ndarray`, `cupy.ndarray`, `torch.Tensor`, etc.) directly. This
is because those are the objects that are going to be passed as inputs to
functions by end users. This does mean that a few behaviors cannot be wrapped
(see below), but most of the array API functional, so this does not affect
most things.

Array consuming library authors coding against the array API may wish to test
against `array_api_strict` to ensure they are not using functionality outside
of the standard, but prefer this implementation for the default behavior for
end-users.

(vendoring)=
## Vendoring

This library supports vendoring as an installation method. To vendor the
library, simply copy `array_api_compat` into the appropriate place in the
library, like

```
cp -R array_api_compat/ mylib/vendored/array_api_compat
```

You may also rename it to something else if you like (nowhere in the code
references the name "array_api_compat").

Alternatively, the library may be installed as dependency from PyPI.

(scope)=
## Scope

At this time, the scope of array-api-compat is limited to wrapping array
libraries so that they can comply with the [array API
standard](https://data-apis.org/array-api/latest/API_specification/index.html).
This includes a small set of [helper functions](helper-functions.rst) which may
be useful to most users of array-api-compat, for instance, functions that
provide meta-functionality to aid in supporting the array API, or functions
that are necessary to work around wrapping limitations for certain libraries.

Things that are out-of-scope include:

- functions that have not yet been
standardized (although note that functions that are in a draft version of the
standard are *in scope*),

- functions that are complicated to implement correctly/maintain,

- anything that requires the use of non-Python code.

If you want a function that is not in array-api-compat that isn't part of the
standard, you should request it either for [inclusion in the
standard](https://github.com/data-apis/array-api/issues) or in specific array
libraries.

Why is the scope limited in this way? Firstly, we want to keep
array-api-compat as primarily a
[polyfill](https://en.wikipedia.org/wiki/Polyfill_(programming)) compatibility
shim. The goal is to let consuming libraries use the array API today, even
with array libraries that do not yet fully support it. In an ideal world---one that we hope to eventually see in the future---array-api-compat would be
unnecessary, because every array library would fully support the standard.

The inclusion of non-standardized functions in array-api-compat would
undermine this goal. But much more importantly, it would also undermine the
goals of the [Data APIs Consortium](https://data-apis.org/). The Consortium
creates the array API standard via the consensus of stakeholders from various
array libraries and users. If a not-yet-standardized function were included in
array-api-compat, it would become *de facto* standard, bypassing the decision
making processes of the Consortium.

Secondly, we want to keep array-api-compat as minimal as possible, so that it
is easy for libraries to add as a (possibly vendored) dependency.

Thirdly, array-api-compat has a relatively small development team. Pull
requests to array-api-compat would not necessarily receive the same stringent
level of scrutiny that changes to established array libraries like NumPy or
PyTorch would. For wrapped standard functions, this is fine, since the
wrappers typically just clean up a few small inconsistencies from the
standard, leaving the complexity of the implementation to the base array
library function. Furthermore, standard functions are tested by the rigorous
[array-api-tests](https://github.com/data-apis/array-api-tests) test suite.
For this reason, functions that require complex implementations are generally
out-of-scope and should be preferred to be implemented in upstream array
libraries.

```{toctree}
:titlesonly:
:hidden:

helper-functions.rst
supported-array-libraries.md
changelog.md
dev/index.md
```
