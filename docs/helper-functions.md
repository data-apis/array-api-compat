# Helper Functions

In addition to the wrapped library namespaces and functions in the array API
specification, there are several helper functions included here that aren't
part of the specification but which are useful for using the array API:

- `is_array_api_obj(x)`: Return `True` if `x` is an array API compatible array
  object.

- `is_numpy_array(x)`, `is_cupy_array(x)`, `is_torch_array(x)`,
  `is_dask_array(x)`, `is_jax_array(x)`: return `True` if `x` is an array from
  the corresponding library. These functions do not import the underlying
  library if it has not already been imported, so they are cheap to use.

- `array_namespace(*xs)`: Get the corresponding array API namespace for the
  arrays `xs`. For example, if the arrays are NumPy arrays, the returned
  namespace will be `array_api_compat.numpy`. Note that this function will
  also work for namespaces that aren't supported by this compat library but
  which do support the array API (i.e., arrays that have the
  `__array_namespace__` attribute).

- `device(x)`: Equivalent to
  [`x.device`](https://data-apis.org/array-api/latest/API_specification/generated/signatures.array_object.array.device.html)
  in the array API specification. Included because `numpy.ndarray` does not
  include the `device` attribute and this library does not wrap or extend the
  array object. Note that for NumPy and dask, `device(x)` is always `"cpu"`.

- `to_device(x, device, /, *, stream=None)`: Equivalent to
  [`x.to_device`](https://data-apis.org/array-api/latest/API_specification/generated/signatures.array_object.array.to_device.html).
  Included because neither NumPy's, CuPy's, Dask's, nor PyTorch's array objects
  include this method. For NumPy, this function effectively does nothing since
  the only supported device is the CPU, but for CuPy, this method supports
  CuPy CUDA
  [Device](https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.Device.html)
  and
  [Stream](https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.Stream.html)
  objects. For PyTorch, this is the same as
  [`x.to(device)`](https://pytorch.org/docs/stable/generated/torch.Tensor.to.html)
  (the `stream` argument is not supported in PyTorch).

- `size(x)`: Equivalent to
  [`x.size`](https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.size.html#array_api.array.size),
  i.e., the number of elements in the array. Included because PyTorch's
  `Tensor` defines `size` as a method which returns the shape, and this cannot
  be wrapped because this compat library doesn't wrap or extend the array
  objects.
