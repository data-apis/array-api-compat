from array_api_compat import (  # noqa: F401
    is_numpy_array, is_cupy_array, is_torch_array,
    is_dask_array, is_jax_array, is_pydata_sparse_array,
    is_numpy_namespace, is_cupy_namespace, is_torch_namespace,
    is_dask_namespace, is_jax_namespace, is_pydata_sparse_namespace,
)

from array_api_compat import is_array_api_obj, device, to_device

from ._helpers import import_, wrapped_libraries, all_libraries

import pytest
import numpy as np
import array
from numpy.testing import assert_allclose

is_array_functions = {
    'numpy': 'is_numpy_array',
    'cupy': 'is_cupy_array',
    'torch': 'is_torch_array',
    'dask.array': 'is_dask_array',
    'jax.numpy': 'is_jax_array',
    'sparse': 'is_pydata_sparse_array',
}

is_namespace_functions = {
    'numpy': 'is_numpy_namespace',
    'cupy': 'is_cupy_namespace',
    'torch': 'is_torch_namespace',
    'dask.array': 'is_dask_namespace',
    'jax.numpy': 'is_jax_namespace',
    'sparse': 'is_pydata_sparse_namespace',
}


@pytest.mark.parametrize('library', is_array_functions.keys())
@pytest.mark.parametrize('func', is_array_functions.values())
def test_is_xp_array(library, func):
    lib = import_(library)
    is_func = globals()[func]

    x = lib.asarray([1, 2, 3])

    assert is_func(x) == (func == is_array_functions[library])

    assert is_array_api_obj(x)


@pytest.mark.parametrize('library', is_namespace_functions.keys())
@pytest.mark.parametrize('func', is_namespace_functions.values())
def test_is_xp_namespace(library, func):
    lib = import_(library)
    is_func = globals()[func]

    assert is_func(lib) == (func == is_namespace_functions[library])


@pytest.mark.parametrize("library", all_libraries)
def test_device(library):
    xp = import_(library, wrapper=True)

    # We can't test much for device() and to_device() other than that
    # x.to_device(x.device) works.

    x = xp.asarray([1, 2, 3])
    dev = device(x)

    x2 = to_device(x, dev)
    assert device(x) == device(x2)


@pytest.mark.parametrize("library", wrapped_libraries)
def test_to_device_host(library):
    # different libraries have different semantics
    # for DtoH transfers; ensure that we support a portable
    # shim for common array libs
    # see: https://github.com/scipy/scipy/issues/18286#issuecomment-1527552919
    xp = import_(library, wrapper=True)

    expected = np.array([1, 2, 3])
    x = xp.asarray([1, 2, 3])
    x = to_device(x, "cpu")
    # torch will return a genuine Device object, but
    # the other libs will do something different with
    # a `device(x)` query; however, what's really important
    # here is that we can test portably after calling
    # to_device(x, "cpu") to return to host
    assert_allclose(x, expected)


@pytest.mark.parametrize("target_library", is_array_functions.keys())
@pytest.mark.parametrize("source_library", is_array_functions.keys())
def test_asarray_cross_library(source_library, target_library, request):
    if source_library == "dask.array" and target_library == "torch":
        # Allow rest of test to execute instead of immediately xfailing
        # xref https://github.com/pandas-dev/pandas/issues/38902

        # TODO: remove xfail once
        # https://github.com/dask/dask/issues/8260 is resolved
        request.node.add_marker(pytest.mark.xfail(reason="Bug in dask raising error on conversion"))
    if source_library == "cupy" and target_library != "cupy":
        # cupy explicitly disallows implicit conversions to CPU
        pytest.skip(reason="cupy does not support implicit conversion to CPU")
    elif source_library == "sparse" and target_library != "sparse":
        pytest.skip(reason="`sparse` does not allow implicit densification")
    src_lib = import_(source_library, wrapper=True)
    tgt_lib = import_(target_library, wrapper=True)
    is_tgt_type = globals()[is_array_functions[target_library]]

    a = src_lib.asarray([1, 2, 3])
    b = tgt_lib.asarray(a)

    assert is_tgt_type(b), f"Expected {b} to be a {tgt_lib.ndarray}, but was {type(b)}"

@pytest.mark.parametrize("library", wrapped_libraries)
def test_asarray_copy(library):
    # Note, we have this test here because the test suite currently doesn't
    # test the copy flag to asarray() very rigorously. Once
    # https://github.com/data-apis/array-api-tests/issues/241 is fixed we
    # should be able to delete this.
    xp = import_(library, wrapper=True)
    asarray = xp.asarray
    is_lib_func = globals()[is_array_functions[library]]
    all = xp.all if library != 'dask.array' else lambda x: xp.all(x).compute()

    if library == 'numpy' and xp.__version__[0] < '2' and not hasattr(xp, '_CopyMode') :
        supports_copy_false = False
    elif library in ['cupy', 'dask.array']:
        supports_copy_false = False
    else:
        supports_copy_false = True

    a = asarray([1])
    b = asarray(a, copy=True)
    assert is_lib_func(b)
    a[0] = 0
    assert all(b[0] == 1)
    assert all(a[0] == 0)

    a = asarray([1])
    if supports_copy_false:
        b = asarray(a, copy=False)
        assert is_lib_func(b)
        a[0] = 0
        assert all(b[0] == 0)
    else:
        pytest.raises(NotImplementedError, lambda: asarray(a, copy=False))

    a = asarray([1])
    if supports_copy_false:
        pytest.raises(ValueError, lambda: asarray(a, copy=False,
                                                  dtype=xp.float64))
    else:
        pytest.raises(NotImplementedError, lambda: asarray(a, copy=False, dtype=xp.float64))

    a = asarray([1])
    b = asarray(a, copy=None)
    assert is_lib_func(b)
    a[0] = 0
    assert all(b[0] == 0)

    a = asarray([1.0], dtype=xp.float32)
    assert a.dtype == xp.float32
    b = asarray(a, dtype=xp.float64, copy=None)
    assert is_lib_func(b)
    assert b.dtype == xp.float64
    a[0] = 0.0
    assert all(b[0] == 1.0)

    a = asarray([1.0], dtype=xp.float64)
    assert a.dtype == xp.float64
    b = asarray(a, dtype=xp.float64, copy=None)
    assert is_lib_func(b)
    assert b.dtype == xp.float64
    a[0] = 0.0
    assert all(b[0] == 0.0)

    # Python built-in types
    for obj in [True, 0, 0.0, 0j, [0], [[0]]]:
        asarray(obj, copy=True) # No error
        asarray(obj, copy=None) # No error
        if supports_copy_false:
            pytest.raises(ValueError, lambda: asarray(obj, copy=False))
        else:
            pytest.raises(NotImplementedError, lambda: asarray(obj, copy=False))

    # Use the standard library array to test the buffer protocol
    a = array.array('f', [1.0])
    b = asarray(a, copy=True)
    assert is_lib_func(b)
    a[0] = 0.0
    assert all(b[0] == 1.0)

    a = array.array('f', [1.0])
    if supports_copy_false:
        b = asarray(a, copy=False)
        assert is_lib_func(b)
        a[0] = 0.0
        assert all(b[0] == 0.0)
    else:
        pytest.raises(NotImplementedError, lambda: asarray(a, copy=False))

    a = array.array('f', [1.0])
    b = asarray(a, copy=None)
    assert is_lib_func(b)
    a[0] = 0.0
    if library == 'cupy':
        # A copy is required for libraries where the default device is not CPU
        assert all(b[0] == 1.0)
    else:
        assert all(b[0] == 0.0)
