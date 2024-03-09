Helper Functions
================

.. currentmodule:: array_api_compat

In addition to the wrapped library namespaces and functions in the array API
specification, there are several helper functions included here that aren't
part of the specification but which are useful for using the array API:

array_namespace
---------------

The `array_namespace()` function is the primary entry-point for array API
consuming libraries.

.. autofunction:: array_namespace
   :canonical: array_api_compat.array_namespace

Array Method Helpers
--------------------

array-api-compat does not attempt to wrap or monkey patch the array object for
any library. Consequently, any API differences for the [array
object](https://data-apis.org/array-api/latest/API_specification/array_object.htmlK
cannot be directly wrapped. Some libraries do not define some of these methods
or define them differently. For these, helper functions are provided which can
be used instead.

Note that if you have a compatibility issue with an operator method (like
`__add__`, i.e., `+`) you can prefer to use the corresponding [elementwise
function](https://data-apis.org/array-api/latest/API_specification/elementwise_functions.html)
instead, which would be wrapped.

.. autofunction:: device
.. autofunction:: to_device
.. autofunction:: size

Inspect Helpers
---------------

These convenience functions can be used to test if an array comes from a
specific library without importing that library if it hasn't been imported
yet.

.. autofunction:: is_array_api_obj
.. autofunction:: is_numpy_array
.. autofunction:: is_cupy_array
.. autofunction:: is_torch_array
.. autofunction:: is_dask_array
.. autofunction:: is_jax_array
