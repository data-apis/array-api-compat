# 1.1.1 (2023-03-08)

## Major Changes

- Rename `get_namespace()` to `array_namespace()` (`get_namespace()` is
  maintained as a backwards compatible alias).

## Minor Changes

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

# 1.1 (2023-02-24)

## Major Changes

- Added support for PyTorch.

- Add helper function `size()` (required if torch is used as
  `torch.Tensor.size` is a method that is incompatible with the array API
  [`.size`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.size.html#array_api.array.size)).

- All wrapper functions that wrap existing library functions now pass through
  arbitrary `**kwargs`.

## Minor Changes

- Added CI to run against the [array API testsuite](https://github.com/data-apis/array-api-tests).

- Fix `sort(stable=False)` and `argsort(stable=False)` with CuPy.

# 1.0 (2022-12-05)

## Major Changes

- Initial release. Includes support for NumPy and CuPy.
