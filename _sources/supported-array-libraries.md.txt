# Supported Array Libraries

The following array libraries are supported. This page outlines the known
differences between this library and the array API specification for the
supported packages.

Note that the {func}`~.array_namespace()` helper will also support any array
library that explicitly supports the array API by defining
[`__array_namespace__`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__array_namespace__.html).

Any reasonably popular array library is in-scope for array-api-compat,
assuming it is possible to wrap it to support the array API without too much
complexity. If your favorite library is not supported, feel free to open an
[issue or pull request](https://github.com/data-apis/array-api-compat/issues).

## [NumPy](https://numpy.org/) and [CuPy](https://cupy.dev/)

NumPy 2.0 has full array API compatibility. This package is not strictly
necessary for NumPy 2.0 support, but may still be useful for the support of
other libraries, as well as for the [helper functions](helper-functions.rst).

For NumPy 1.26, as well as corresponding versions of CuPy, the following
deviations from the standard should be noted:

- The array methods `__array_namespace__`, `device` (for NumPy), `to_device`,
  and `mT` are not defined. This reuses `np.ndarray` and `cp.ndarray` and we
  don't want to monkey patch or wrap it. The [helper
  functions](helper-functions.rst) {func}`~.device()` and {func}`~.to_device()`
  are provided to work around these missing methods. `x.mT` can be replaced
  with `xp.linalg.matrix_transpose(x)`. {func}`~.array_namespace()` should be
  used instead of `x.__array_namespace__`.

- Value-based casting for scalars will be in effect unless explicitly disabled
  with the environment variable `NPY_PROMOTION_STATE=weak` or
  `np._set_promotion_state('weak')` (requires NumPy 1.24 or newer, see [NEP
  50](https://numpy.org/neps/nep-0050-scalar-promotion.html) and
  https://github.com/numpy/numpy/issues/22341)

- Functions which are not wrapped may not have the same type annotations
  as the spec.

- Functions which are not wrapped may not use positional-only arguments.

The minimum supported NumPy version is 1.22. However, this older version of
NumPy has a few issues:

- `unique_*` will not compare nans as unequal.
- No `from_dlpack` or `__dlpack__`.
- Type promotion behavior will be value based for 0-D arrays (and there is no
  `NPY_PROMOTION_STATE=weak` to disable this).

If any of these are an issue, it is recommended to bump your minimum NumPy
version.

## [PyTorch](https://pytorch.org/)

- Like NumPy/CuPy, we do not wrap the `torch.Tensor` object. It is missing the
  `__array_namespace__` and `to_device` methods, so the corresponding helper
  functions {func}`~.array_namespace()` and {func}`~.to_device()` in this
  library should be used instead.

- The {external+torch:meth}`x.size() <torch.Tensor.size>` attribute on
  `torch.Tensor` is a method that behaves differently from the
  [`x.size`](https://data-apis.org/array-api/draft/API_specification/generated/array_api.array.size.html)
  attribute in the spec. Use the {func}`~.size()` helper function as a
  portable workaround.

- PyTorch has incomplete support for unsigned integer types other
  than `uint8`, and no attempt is made to implement them here.

- PyTorch has type promotion semantics that differ from the array API
  specification for 0-D tensor objects. The array functions in this wrapper
  library do work around this, but the operators on the Tensor object do not,
  as no operators or methods on the Tensor object are modified. If this is a
  concern, use the functional form instead of the operator form, e.g., `add(x,
  y)` instead of `x + y`.

- [`unique_all()`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.unique_all.html#array_api.unique_all)
  is not implemented, due to the fact that `torch.unique` does not support
  returning the `indices` array. The other
  [`unique_*`](https://data-apis.org/array-api/latest/API_specification/set_functions.html)
  functions are implemented.

- Slices do not support negative steps.

- [`std()`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.std.html#array_api.std)
  and
  [`var()`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.var.html#array_api.var)
  do not support floating-point `correction`.

- The `stream` argument of the {func}`~.to_device()` helper is not supported.

- As with NumPy, type annotations and positional-only arguments may not
  exactly match the spec for functions that are not wrapped at all.

(jax-support)=
## [JAX](https://jax.readthedocs.io/en/latest/)

Unlike the other libraries supported here, JAX array API support is contained
entirely in the JAX library. The JAX array API support is tracked at
https://github.com/google/jax/issues/18353.

## [Dask](https://www.dask.org/)

If you're using dask with numpy, many of the same limitations that apply to numpy
will also apply to dask. Besides those differences, other limitations include missing
sort functionality (no `sort` or `argsort`), and limited support for the optional `linalg`
and `fft` extensions.

In particular, the `fft` namespace is not compliant with the array API spec. Any functions
that you find under the `fft` namespace are the original, unwrapped functions under [`dask.array.fft`](https://docs.dask.org/en/latest/array-api.html#fast-fourier-transforms), which may or may not be Array API compliant. Use at your own risk!

For `linalg`, several methods are missing, for example:
- `cross`
- `det`
- `eigh`
- `eigvalsh`
- `matrix_power`
- `pinv`
- `slogdet`
- `matrix_norm`
- `matrix_rank`
Other methods may only be partially implemented or return incorrect results at times.

(sparse-support)=
## [Sparse](https://sparse.pydata.org/en/stable/)

Similar to JAX, `sparse` Array API support is contained directly in `sparse`.

(ndonnx-support)=
## [ndonnx](https://github.com/quantco/ndonnx)

Similar to JAX, `ndonnx` Array API support is contained directly in `ndonnx`.

(array-api-strict-support)=
## [array-api-strict](https://data-apis.org/array-api-strict/)

array-api-strict exists only to test support for the Array API, so it does not need any wrappers.
