# Array API compatibility library

This is a small wrapper around common array libraries that is compatible with
the [Array API standard](https://data-apis.org/array-api/latest/). Currently,
NumPy, CuPy, PyTorch, Dask, and JAX are supported. If you want support for other array
libraries, or if you encounter any issues, please [open an
issue](https://github.com/data-apis/array-api-compat/issues).

Note that some of the functionality in this library is backwards incompatible
with the corresponding wrapped libraries. The end-goal is to eventually make
each array library itself fully compatible with the array API, but this
requires making backwards incompatible changes in many cases, so this will
take some time.

Currently all libraries here are implemented against the [2022.12
version](https://data-apis.org/array-api/2022.12/) of the standard.

## Install

`array-api-compat` is available on both [PyPI](https://pypi.org/project/array-api-compat/)

```
python -m pip install array-api-compat
```

and [Conda-forge](https://anaconda.org/conda-forge/array-api-compat)

```
conda install --channel conda-forge array-api-compat
```

## Usage

The typical usage of this library will be to get the corresponding array API
compliant namespace from the input arrays using `array_namespace()`, like

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

> [!NOTE]
> There is no `array_api_compat.jax` submodule. JAX support is contained
> in JAX itself in the `jax.experimental.array_api` module. array-api-compat simply
> wraps that submodule. The main JAX support in this module consists of
> supporting it in the [helper functions](helper-functions) defined below.

Each will include all the functions from the normal NumPy/CuPy/PyTorch/dask.array
namespace, except that functions that are part of the array API are wrapped so
that they have the correct array API behavior. In each case, the array object
used will be the same array object from the wrapped library.

(array-api-strict)=
## Difference between `array_api_compat` and `array_api_strict`

[`array_api_strict`](https://github.com/data-apis/array-api-strict) is a
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

Alternatively, the library may be installed as dependency on PyPI.


```{toctree}
:titlesonly:
:hidden:

helper-functions.md
supported-array-libraries.md
dev/index.md
```
