# Changelog

## 1.5.1 (2024-03-20)

## Minor Changes

- Add [HTML documentation](https://data-apis.org/array-api-compat/). Includes
  new documentation on the [module scope](scope) and new [developer
  documentation](dev/index.md).

- Fix `array_api_compat.numpy.asarray(torch.Tensor)` to return a NumPy array.

- Allow Python scalars in torch functions.

- Fix the `torch.std` wrapper when correction is an `int`.

- Fix issues with `qr` and `svd` in the Dask wrappers.

## 1.5 (2024-03-07)

### Major Changes

- Add support for Dask ([@lithomas1](https://github.com/lithomas1)).

- Add support for JAX. Note that unlike other array libraries,
  array-api-compat does not contain any wrappers for JAX functions. All JAX
  array API support is in JAX itself. Thus, there is no `array_api_compat.jax`
  submodule, and `array_namespace(<JAX array>)` returns the
  `jax.experimental.array_api` module.

- The functions `is_numpy_array(x)`, `is_cupy_array(x)`, `is_torch_array(x)`,
  `is_dask_array(x)`, `is_jax_array(x)` are now part of the public
  `array_api_compat` API.

- Add wrappers for the `fft` extension module for NumPy, CuPy, and PyTorch.

### Minor Changes

- Allow `'2022.12'` as the `api_version` in `array_namespace()`. `'2021.12'`
  is also supported but will issue a warning since the returned namespace will
  still be a 2022.12 compliant one.

- Add wrapper for numpy.linalg.solve, which broadcasts the inputs according to
  the standard.

- Add wrappers for various PyTorch linalg functions.

- Fix a bug with `numpy.linalg.vector_norm(keepdims=True)`.

- BREAKING: Update `vecdot` wrappers to apply `axes` before broadcasting, not
  after. This matches the updated 2023.12 standard wording, and also the
  behavior of the new `numpy.vecdot` gufunc in NumPy 2.0.

- Fix some linalg functions which were supposed to be in both the main
  namespace and the linalg extension namespace.

- Add Ruff to CI. ([@adonath](https://github.com/adonath))

- Test that internal definitions of `__all__` are self-consistent, which
  should help to avoid issues where wrappers are accidentally not exported to
  the compat namespaces properly.

## 1.4.1 (2024-01-18)

### Minor Changes

- Add support for the upcoming NumPy 2.0 release.

- Added a torch wrapper for `trace` (`torch.trace` doesn't support the
  `offset` argument or stacking)

- Wrap numpy, cupy, and torch `nonzero` to raise an error for zero-dimensional
  input arrays.

- Add torch wrapper for `newaxis`.

- Improve error message for `array_namespace`

- Fix linalg.cholesky returning the conjugate of the expected upper
  decomposition for numpy and cupy.

## 1.4 (2023-09-13)

### Major Changes

- Releases are now made with GitHub Actions (thanks
  [@matthewfeickert](https://github.com/matthewfeickert)).

### Minor Changes

- Fix `torch.result_type()` cross-kind promotion
  ([@lucascolley](https://github.com/lucascolley)).

- Fix the torch.take() wrapper to make axis optional for ndim = 1.

- Add requires-python metadata to the package
  ([@matthewfeickert](https://github.com/matthewfeickert)).

## 1.3 (2023-06-20)

### Major Changes

- Add [2022.12](https://data-apis.org/array-api/2022.12/) standard support.
  This includes things like adding complex dtype support, adding the new
  `take` function, and various minor changes in the specification.

### Minor Changes

- Support `"cpu"` in CuPy `to_device()`.

- Return a new array in NumPy/CuPy `reshape(copy=False)`.

- Fix signatures for PyTorch `broadcast_to` and `permute_dims`.

## 1.2 (2023-04-03)

### Major Changes

- Support the linalg extension in the `array_api_compat.torch` namespace.

- Add `isdtype()`.

### Minor Changes

- Fix the `k` keyword argument to `tril` and `triu` in `torch`.

## 1.1.1 (2023-03-10)

### Major Changes

- Rename `get_namespace()` to `array_namespace()` (`get_namespace()` is
  maintained as a backwards compatible alias).

### Minor Changes

- The minimum supported NumPy version is now 1.21. Fixed a few issues with
  NumPy 1.21 (with `unique_*` and `asarray`), although there are also a few
  known issues with this version (see the README).

- Add `api_version` to `get_namespace()`.

- `array_namespace()` (*née* `get_namespace()`) now works correctly with
  `torch` tensors.

- `array_namespace()` (*née* `get_namespace()`) now works correctly with
  `numpy.array_api` arrays.

- `array_namespace()` (*née* `get_namespace()`) now raises `TypeError` instead
  of `ValueError`.

- Fix the `torch.std` wrapper.

- Add `torch` wrappers for `ones`, `empty`, and `zeros` so that `shape` can be
  passed as a keyword argument.

## 1.1 (2023-02-24)

### Major Changes

- Added support for PyTorch.

- Add helper function `size()` (required if torch is used as
  `torch.Tensor.size` is a method that is incompatible with the array API
  [`.size`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.size.html#array_api.array.size)).

- All wrapper functions that wrap existing library functions now pass through
  arbitrary `**kwargs`.

### Minor Changes

- Added CI to run against the [array API testsuite](https://github.com/data-apis/array-api-tests).

- Fix `sort(stable=False)` and `argsort(stable=False)` with CuPy.

## 1.0 (2022-12-05)

### Major Changes

- Initial release. Includes support for NumPy and CuPy.
