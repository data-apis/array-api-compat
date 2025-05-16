# Changelog

## 1.12.0 (2025-05-13)


### Major changes

- The build system has been updated to use `pyproject.toml` instead of `setup.py`
- Support for Python 3.9 has been dropped. The minimum supported Python version is now
  3.10; the minimum supported NumPy version is 1.22.
- The `linalg` extension works correctly with `pytorch>=2.7`.
- Multiple improvements to handling of devices in CuPy and PyTorch backends.
  Support for multiple devices in CuPy is still immature and you should use
  context managers rather than relying on input-output device propagation or
  on the `device` parameter.  Please report any issues you encounter.

### Minor changes

- `finfo` and `iinfo` functions now accept array arguments, in accordance with the
   Array API spec;
- `torch.asarray` function propagates the device of the input array. This works around
   the [pytorch issue #150199](https://github.com/pytorch/pytorch/issues/150199);
- `torch.repeat` function is now available;
- `torch.count_nonzero` function now correctly handles the case of a tuple `axis`
  arguments and `keepdims=True`;
- `torch.meshgrid` wrapper defaults to `indexing="xy"`, in accordance with the
  array API specification;
- `cupy.asarray` function now implements the `copy=False` argument, albeit
  at the cost of risking to make a temporary copy.
- In `numpy.take_along_axis` and `cupy.take_along_axis` the `axis` parameter now
  defaults to -1, in accordance to the Array API spec.


The following users contributed to this release:

Evgeni Burovski,
Lucas Colley,
Neil Girdhar,
Joren Hammudoglu,
Guido Imperiale


## 1.11.2 (2025-03-20)

This is a bugfix release with no new features compared to version 1.11.

- fix the `result_type` wrapper for pytorch. Previously, `result_type` had multiple
  issues with scalar arguments.
- fix several issues with `clip` wrappers. Previously, `clip` was failing to allow
  behaviors which are unspecified by the 2024.12 standard but allowed by the array
  libraries.

The following users contributed to this release:

Evgeni Burovski
Guido Imperiale
Magnus Dalen Kvalevåg


## 1.11.1 (2025-03-04)

This is a bugfix release with no new features compared to version 1.11.

### Major Changes

- fix `count_nonzero` wrappers: work around the lack of the `keepdims` argument in
  several array libraries (torch, dask, cupy); work around numpy returning python
  ints in for some input combinations.

### Minor Changes

- runnings self-tests does not require all array libraries. Missing libraries are
  skipped.

The following users contributed to this release:

Evgeni Burovski
Guido Imperiale


## 1.11.0 (2025-02-27)

### Major Changes

This release targets the 2024.12 Array API revision. This includes

  - `__array_api_version__` for the wrapped APIs is now set to `2024.12`;
  - Wrappers for `count_nonzero`;
  - Wrappers for `cumulative_prod`;
  - Wrappers for `take_along_axis` (with the exception of Dask);
  - Wrappers for `diff`;
  - `__capabilities__` dict contains a `max_dimensions` key;
  - Python scalars are accepted as arguments to `result_type`;
  - `fft.fftfreq` and `fft.rfftfreq` functions now accept an optional `dtype`
    argument to control the output data type.

Note that these wrappers, as well as other 2024.12 features, are relatively undertested
in this release, and may have rough edges. Please report any issues you encounter
in [the issue tracker](https://github.com/data-apis/array-api-compat/issues).

New functions to test properties of arrays:
  - `is_writeable_array` (benefits NumPy, JAX, Sparse)
  - `is_lazy_array` (benefits JAX, Dask, ndonnx)

Improved support for JAX:
  - Work arounds for `.device` attribute and `to_device` function
    not working correctly within `jax.jit`

### Minor Changes

- Several improvements to `dask.array` wrappers:

  - `size` returns None for arrays of unknown shapes.
  - `astype(..., copy=True)` always copies, independently of the Dask version.
  - implementations of `sort` and `argsort` are now available. Note that these
    implementations are relatively crude, and might be memory intensive.
  - `asarray` no longer accidentally materializes the Dask graph
  - `torch` wrappers contain unsigned integer dtypes of widths >8 bits, `uint16`,
    `uint32` and `uint64` if PyTorch version is at least 2.3. Note that the
     unsigned integer support is incomplete in PyTorch itself, see
     [gh-253](https://github.com/data-apis/array-api-compat/pull/253).

### Authors

The following users contributed to this release:

Athan Reines
Guido Imperiale
Evgeni Burovski
Guido Imperiale
Lucas Colley
Ralf Gommers
Thomas Li


## 1.10.0 (2024-12-25)

### Major Changes

- New function `is_writeable_array` adds transparent support for readonly
  arrays, such as JAX arrays or numpy arrays with `.flags.writeable=False`.

- `asarray(..., copy=None)` with `dask` backend always copies, so that
  `copy=None` and `copy=True` are equivalent for the `dask` backend.
   This change is made to be forward compatible with the `dask==2024.12`
   release.


### Minor Changes

- `array_namespace` accepts (and ignores) `None` and python scalars (int, float,
   complex, bool). This change is to simplify downstream adoption, for
   functions where arguments can be either arrays or scalars.

- `vecdot` conjugates its first argument, as stipulated by the Array API spec.
  Previously, conjation if the first argument was missing.


## 1.9.1 (2024-10-29)

### Major Changes

- `__array_api_version__` for the wrapped APIs is now set to `2023.12`.

### Minor Changes

- Wrap `sign` so that it always uses the standard definition for complex
  numbers, and always propagates nans.

- Wrap dask.array.fft.

- Readd `python_requires` to the package metadata.

## 1.9 (2024-10-??)

### Major Changes

- New helper functions to determine if a namespace is from a given library
  ({func}`~.is_numpy_namespace`, {func}`~.is_torch_namespace`, etc.).

- More support for the [2023.12 version of the
  standard](https://data-apis.org/array-api/latest/changelog.html#v2023-12).
  This includes
  - Wrappers for `cumulative_sum()`.
  - Wrappers for `unstack()`.
  - Update floating-point type promotion in `sum()`, `prod()`, and `trace()`
    to be inline with the 2023.12 specification (32-bit types no longer
    promote to 64-bit when `dtype=None`).
  - Add the [inspection
    APIs](https://data-apis.org/array-api/latest/API_specification/inspection.html)
    to the wrapped namespaces. These can be accessed with
    `xp.__array_namespace_info__()`.
  - Various fixes to the `clip()` wrappers.

- `torch.conj` now wrapps `torch.conj_physical`, which makes a copy rather
  than setting the conjugation bit, as arrays with the conjugation bit set do
  not support some APIs.

- `torch.sign` is now wrapped to support complex numbers and propogate nans
  properly.

### Minor Changes

- NumPy 2.0 is now wrapped again. Previously it was unwrapped because it has
  full 2022.12 array API support but it now requires wrapping again for
  2023.12 support.

- Support for JAX 0.4.32 and newer which implements the array API directly in
  `jax.numpy`.

- `hypot`, `minimum`, and `maximum` (new in 2023.12) are wrapped in PyTorch to
  support proper scalar type promotion.

## 1.8 (2024-07-24)

### Major Changes

- Add support for [ndonnx](https://github.com/Quantco/ndonnx). Array API
  support itself lives in the ndonnx library, but this adds the
  {func}`~.is_ndonnx_array` helper function.
  ([@adityagoel4512](https://github.com/adityagoel4512)).

- Partial support for the [2023.12 version of the
  standard](https://data-apis.org/array-api/latest/changelog.html#v2023-12).
  This includes
  - Wrappers for `clip()`.
  - torch wrapper for `copysign()` with correct type promotion.

  Note that many of the new functions in the 2023.12 version of the standard
  are already fully implemented in upstream libraries and will already work.

## 1.7.1 (2024-05-28)

### Minor Changes

- Fix a typo in setup.py ([@sunpoet](https://github.com/sunpoet)).

## 1.7 (2024-05-24)

### Major Changes

- Add support for `sparse`. Note that unlike other array libraries,
  array-api-compat does not contain any wrappers for `sparse` functions. All
  `sparse` array API support is in `sparse` itself. Thus, there is no
  `array_api_compat.sparse` submodule, and
  `array_namespace(<pydata/sparse array>)` returns the `sparse` module.

- Added the function `is_pydata_sparse_array(x)`.

### Minor Changes

- Fix JAX `float0` arrays. See https://github.com/google/jax/issues/20620.
  ([@NeilGirdhar](https://github.com/NeilGirdhar))

- Fix `torch.linalg.vector_norm()` when `axis=()`.

- Fix `torch.linalg.solve()` to apply the array API standard rules for when
  `x2` should be treated as a vector vs. a matrix.

- Fix PyTorch test failures on CI by skipping uint16, uint32, uint64 tests.

## 1.6 (2024-03-29)

### Major Changes

- Drop support for Python 3.8.

- NumPy 2.0 is now left completely unwrapped.

- New flag `use_compat` to {func}`~.array_namespace` to force the use or
  non-use of the compat wrapper namespace. The default is to return a compat
  namespace when it is appropiate.

- Fix the `copy` flag to `asarray` for NumPy, CuPy, and Dask.

- Fix the `device` flag to `asarray` for CuPy.

- Fix various issues with `asarray` for Dask.

### Minor Changes

- Test Python 3.12 on CI.

- Add more tests for {func}`~.array_namespace`.

- Add more tests for `asarray`.

- Add a test that there are no hard dependencies.

## 1.5.1 (2024-03-20)

### Minor Changes

- Add [HTML documentation](https://data-apis.org/array-api-compat/). Includes
  new documentation on the [scope of the package](scope) and new [developer
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

- Allow `'2022.12'` as the `api_version` in {func}`~.array_namespace()`.
  `'2021.12'` is also supported but will issue a warning since the returned
  namespace will still be a 2022.12 compliant one.

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
