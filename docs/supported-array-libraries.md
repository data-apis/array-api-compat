# Supported Array Libraries

The following array libraries are supported. This page outlines the known
differences between this library and the array API specification for the
supported packages.

Note that the [`array_namespace()`](helper-functions.md) helper will also
support any array library that explicitly supports the array API by defining
[`__array_namespace__`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__array_namespace__.html).

## [NumPy](https://numpy.org/) and [CuPy](https://cupy.dev/)

NumPy 2.0 has full array API compatibility. This package is not strictly
necessary for NumPy 2.0 support, but may still be useful for the support of
other libraries, as well as for the [helper functions](helper-functions.md).

For NumPy 1.26, as well as corresponding versions of CuPy, the following
deviations from the standard should be noted:

- The array methods `__array_namespace__`, `device` (for NumPy), `to_device`,
  and `mT` are not defined. This reuses `np.ndarray` and `cp.ndarray` and we
  don't want to monkey patch or wrap it. The [helper
  functions](helper-functions.md) `device()` and `to_device()` are provided to
  work around these missing methods. `x.mT` can be replaced with
  `xp.linalg.matrix_transpose(x)`. `array_namespace(x)` should be used instead
  of `x.__array_namespace__`.

- Value-based casting for scalars will be in effect unless explicitly disabled
  with the environment variable `NPY_PROMOTION_STATE=weak` or
  `np._set_promotion_state('weak')` (requires NumPy 1.24 or newer, see [NEP
  50](https://numpy.org/neps/nep-0050-scalar-promotion.html) and
  https://github.com/numpy/numpy/issues/22341)

- `asarray()` does not support `copy=False`.

- Functions which are not wrapped may not have the same type annotations
  as the spec.

- Functions which are not wrapped may not use positional-only arguments.

The minimum supported NumPy version is 1.21. However, this older version of
NumPy has a few issues:

- `unique_*` will not compare nans as unequal.
- `finfo()` has no `smallest_normal`.
- No `from_dlpack` or `__dlpack__`.
- `argmax()` and `argmin()` do not have `keepdims`.
- `qr()` doesn't support matrix stacks.
- `asarray()` doesn't support `copy=True` (as noted above, `copy=False` is not
  supported even in the latest NumPy).
- Type promotion behavior will be value based for 0-D arrays (and there is no
  `NPY_PROMOTION_STATE=weak` to disable this).

If any of these are an issue, it is recommended to bump your minimum NumPy
version.

## [PyTorch](https://pytorch.org/)

- Like NumPy/CuPy, we do not wrap the `torch.Tensor` object. It is missing the
  `__array_namespace__` and `to_device` methods, so the corresponding helper
  functions `array_namespace()` and `to_device()` in this library should be
  used instead (see above).

- The `x.size` attribute on `torch.Tensor` is a function that behaves
  differently from
  [`x.size`](https://data-apis.org/array-api/draft/API_specification/generated/array_api.array.size.html)
  in the spec. Use the `size(x)` helper function as a portable workaround (see
  above).

- PyTorch does not have unsigned integer types other than `uint8`, and no
  attempt is made to implement them here.

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

- The `stream` argument of the `to_device()` helper (see above) is not
  supported.

- As with NumPy, type annotations and positional-only arguments may not
  exactly match the spec for functions that are not wrapped at all.

The minimum supported PyTorch version is 1.13.

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

The minimum supported Dask version is 2023.12.0.
