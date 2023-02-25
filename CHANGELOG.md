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
