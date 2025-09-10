Helper Functions
================

.. currentmodule:: array_api_compat

In addition to the wrapped library namespaces and functions in the array API
specification, there are several helper functions included here that aren't
part of the specification but which are useful for using the array API:

Entry-point Helpers
-------------------

The `array_namespace()` function is the primary entry-point for array API
consuming libraries.


.. autofunction:: array_namespace
.. autofunction:: is_array_api_obj

Array Method Helpers
--------------------

array-api-compat does not attempt to wrap or monkey patch the array object for
any library. Consequently, any API differences for the `array object
<https://data-apis.org/array-api/latest/API_specification/array_object.html>`__
cannot be directly wrapped. Some libraries do not define some of these methods
or define them differently. For these, helper functions are provided which can
be used instead.

Note that if you have a compatibility issue with an operator method (like
`__add__`, i.e., `+`) you can prefer to use the corresponding `elementwise
function
<https://data-apis.org/array-api/latest/API_specification/elementwise_functions.html>`__
instead, which would be wrapped.

.. autofunction:: device
.. autofunction:: to_device
.. autofunction:: size

Inspection Helpers
------------------

These convenience functions can be used to test if an array or namespace comes from a
specific library without importing that library if it hasn't been imported
yet.

.. autofunction:: is_numpy_array
.. autofunction:: is_cupy_array
.. autofunction:: is_torch_array
.. autofunction:: is_dask_array
.. autofunction:: is_jax_array
.. autofunction:: is_pydata_sparse_array
.. autofunction:: is_ndonnx_array
.. autofunction:: is_writeable_array
.. autofunction:: is_lazy_array
.. autofunction:: is_numpy_namespace
.. autofunction:: is_cupy_namespace
.. autofunction:: is_torch_namespace
.. autofunction:: is_dask_namespace
.. autofunction:: is_jax_namespace
.. autofunction:: is_pydata_sparse_namespace
.. autofunction:: is_ndonnx_namespace
.. autofunction:: is_array_api_strict_namespace
