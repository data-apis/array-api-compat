import math

import pytest
import numpy as np
import array
from numpy.testing import assert_equal

from array_api_compat import (  # noqa: F401
    is_numpy_array, is_cupy_array, is_torch_array,
    is_dask_array, is_jax_array, is_pydata_sparse_array,
    is_ndonnx_array,
    is_numpy_namespace, is_cupy_namespace, is_torch_namespace,
    is_dask_namespace, is_jax_namespace, is_pydata_sparse_namespace,
    is_array_api_strict_namespace, is_ndonnx_namespace,
)

from array_api_compat import (
    device, is_array_api_obj, is_lazy_array, is_writeable_array, size, to_device
)
from array_api_compat.common._helpers import _DASK_DEVICE
from ._helpers import all_libraries, import_, wrapped_libraries, xfail


is_array_functions = {
    'numpy': 'is_numpy_array',
    'cupy': 'is_cupy_array',
    'torch': 'is_torch_array',
    'dask.array': 'is_dask_array',
    'jax.numpy': 'is_jax_array',
    'sparse': 'is_pydata_sparse_array',
    'ndonnx': 'is_ndonnx_array',
}

is_namespace_functions = {
    'numpy': 'is_numpy_namespace',
    'cupy': 'is_cupy_namespace',
    'torch': 'is_torch_namespace',
    'dask.array': 'is_dask_namespace',
    'jax.numpy': 'is_jax_namespace',
    'sparse': 'is_pydata_sparse_namespace',
    'array_api_strict': 'is_array_api_strict_namespace',
    'ndonnx': 'is_ndonnx_namespace',
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


@pytest.mark.parametrize('library', all_libraries)
def test_xp_is_array_generics(library):
    """
    Test that scalar selection on a xp.ndarray always returns
    an object that matches with exactly one among the is_*_array
    function of the same library and is_numpy_array.
    """
    lib = import_(library)
    x = lib.asarray([1, 2, 3])
    x0 = x[0]

    matches = []
    for library2, func in is_array_functions.items():
        is_func = globals()[func]
        if is_func(x0):
            matches.append(library2)

    if library == "array_api_strict":
        # There is no is_array_api_strict_array() function
        assert matches == []
    else:
        assert matches in ([library], ["numpy"])


@pytest.mark.parametrize("library", all_libraries)
def test_is_writeable_array(library):
    lib = import_(library)
    x = lib.asarray([1, 2, 3])
    if is_writeable_array(x):
        x[1] = 4
    else:
        with pytest.raises((TypeError, ValueError)):
            x[1] = 4


def test_is_writeable_array_numpy():
    x = np.asarray([1, 2, 3])
    assert is_writeable_array(x)
    x.flags.writeable = False
    assert not is_writeable_array(x)


@pytest.mark.parametrize("library", all_libraries)
def test_size(library):
    xp = import_(library)
    x = xp.asarray([1, 2, 3])
    assert size(x) == 3


@pytest.mark.parametrize("library", all_libraries)
def test_size_none(library):
    if library == "sparse":
        pytest.skip("No arange(); no indexing by sparse arrays")

    xp = import_(library)
    x = xp.arange(10)
    x = x[x < 5]

    # dask.array now has shape=(nan, ) and size=nan
    # ndonnx now has shape=(None, ) and size=None
    # Eager libraries have shape=(5, ) and size=5
    assert size(x) in (None, 5)


@pytest.mark.parametrize("library", all_libraries)
def test_is_lazy_array(library):
    lib = import_(library)
    x = lib.asarray([1, 2, 3])
    assert isinstance(is_lazy_array(x), bool)


@pytest.mark.parametrize("shape", [(math.nan,), (1, math.nan), (None, ), (1, None)])
def test_is_lazy_array_nan_size(shape, monkeypatch):
    """Test is_lazy_array() on an unknown Array API compliant object
    with NaN (like Dask) or None (like ndonnx) in its shape
    """
    xp = import_("array_api_strict")
    x = xp.asarray(1)
    assert not is_lazy_array(x)
    monkeypatch.setattr(type(x), "shape", shape)
    assert is_lazy_array(x)


@pytest.mark.parametrize("exc", [TypeError, AssertionError])
def test_is_lazy_array_bool_raises(exc, monkeypatch):
    """Test is_lazy_array() on an unknown Array API compliant object
    where calling bool() raises:
    - TypeError: e.g. like jitted JAX. This is the proper exception which
      lazy arrays should raise as per the Array API specification
    - something else: e.g. like Dask, where bool() triggers compute()
      which can result in any kind of exception to be raised
    """
    xp = import_("array_api_strict")
    x = xp.asarray(1)
    assert not is_lazy_array(x)

    def __bool__(self):
        raise exc("Hello world")

    monkeypatch.setattr(type(x), "__bool__", __bool__)
    assert is_lazy_array(x)


@pytest.mark.parametrize(
    'func',
    list(is_array_functions.values()) 
    + ["is_array_api_obj", "is_lazy_array", "is_writeable_array"]
)
def test_is_array_any_object(func):
    """Test that is_*_array functions return False and don't raise on non-array objects
    """
    func = globals()[func]

    # These objects are missing attributes such as __name__
    assert not func(object())
    assert not func(None)
    assert not func(1)

    class C:
        pass

    assert not func(C())


@pytest.mark.parametrize("library", all_libraries)
def test_device_to_device(library, request):
    if library == "ndonnx":
        xfail(request, reason="Stub raises ValueError")
    if library == "sparse":
        xfail(request, reason="No __array_namespace_info__()")

    xp = import_(library, wrapper=True)
    devices = xp.__array_namespace_info__().devices()

    # Default device
    x = xp.asarray([1, 2, 3])
    dev = device(x)

    for dev in devices:
        if dev is None:  # JAX >=0.5.3
            continue
        if dev is _DASK_DEVICE:  # TODO this needs a better design
            continue
        y = to_device(x, dev)
        assert device(y) == dev


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
    assert_equal(x, expected)


@pytest.mark.parametrize("target_library", is_array_functions.keys())
@pytest.mark.parametrize("source_library", is_array_functions.keys())
def test_asarray_cross_library(source_library, target_library, request):
    if source_library == "dask.array" and target_library == "torch":
        # TODO: remove xfail once
        # https://github.com/dask/dask/issues/8260 is resolved
        xfail(request, reason="Bug in dask raising error on conversion")

    elif (
        source_library == "ndonnx" 
        and target_library not in ("array_api_strict", "ndonnx", "numpy")
    ):
        xfail(request, reason="The truth value of lazy Array Array(dtype=Boolean) is unknown")
    elif source_library == "ndonnx" and target_library == "numpy":
        xfail(request, reason="produces numpy array of ndonnx scalar arrays")
    elif target_library == "ndonnx" and source_library in ("torch", "dask.array", "jax.numpy"):
        xfail(request, reason="unable to infer dtype")

    elif source_library == "jax.numpy" and target_library == "torch":
        xfail(request, reason="casts int to float")
    elif source_library == "cupy" and target_library != "cupy":
        # cupy explicitly disallows implicit conversions to CPU
        pytest.skip(reason="cupy does not support implicit conversion to CPU")
    elif source_library == "sparse" and target_library != "sparse":
        pytest.skip(reason="`sparse` does not allow implicit densification")

    src_lib = import_(source_library, wrapper=True)
    tgt_lib = import_(target_library, wrapper=True)
    is_tgt_type = globals()[is_array_functions[target_library]]

    a = src_lib.asarray([1, 2, 3], dtype=src_lib.int32)
    b = tgt_lib.asarray(a)

    assert is_tgt_type(b), f"Expected {b} to be a {tgt_lib.ndarray}, but was {type(b)}"
    assert b.dtype == tgt_lib.int32


@pytest.mark.parametrize("library", wrapped_libraries)
def test_asarray_copy(library):
    # Note, we have this test here because the test suite currently doesn't
    # test the copy flag to asarray() very rigorously. Once
    # https://github.com/data-apis/array-api-tests/issues/241 is fixed we
    # should be able to delete this.
    xp = import_(library, wrapper=True)
    asarray = xp.asarray
    is_lib_func = globals()[is_array_functions[library]]

    a = asarray([1])
    b = asarray(a, copy=True)
    assert is_lib_func(b)
    a[0] = 0
    assert b[0] == 1
    assert a[0] == 0

    a = asarray([1])

    # Test copy=False within the same namespace
    b = asarray(a, copy=False)
    assert is_lib_func(b)
    a[0] = 0
    assert b[0] == 0
    with pytest.raises(ValueError):
        asarray(a, copy=False, dtype=xp.float64)

    # copy=None defaults to False when possible
    a = asarray([1])
    b = asarray(a, copy=None)
    assert is_lib_func(b)
    a[0] = 0
    assert b[0] == 0

    # copy=None defaults to True when impossible
    a = asarray([1.0], dtype=xp.float32)
    assert a.dtype == xp.float32
    b = asarray(a, dtype=xp.float64, copy=None)
    assert is_lib_func(b)
    assert b.dtype == xp.float64
    a[0] = 0.0
    assert b[0] == 1.0

    # copy=None defaults to False when possible
    a = asarray([1.0], dtype=xp.float64)
    assert a.dtype == xp.float64
    b = asarray(a, dtype=xp.float64, copy=None)
    assert is_lib_func(b)
    assert b.dtype == xp.float64
    a[0] = 0.0
    assert b[0] == 0.0

    # Python built-in types
    for obj in [True, 0, 0.0, 0j, [0], [[0]]]:
        asarray(obj, copy=True)  # No error
        asarray(obj, copy=None)  # No error

        with pytest.raises(ValueError):
            asarray(obj, copy=False)

    # Use the standard library array to test the buffer protocol
    a = array.array("f", [1.0])
    b = asarray(a, copy=True)
    assert is_lib_func(b)
    a[0] = 0.0
    assert b[0] == 1.0

    a = array.array("f", [1.0])
    if library in ("cupy", "dask.array"):
        with pytest.raises(ValueError):
            asarray(a, copy=False)
    else:
        b = asarray(a, copy=False)
        assert is_lib_func(b)
        a[0] = 0.0
        assert b[0] == 0.0

    a = array.array("f", [1.0])
    b = asarray(a, copy=None)
    assert is_lib_func(b)
    a[0] = 0.0
    if library in ("cupy", "dask.array"):
        # A copy is required for libraries where the default device is not CPU
        # dask changed behaviour of copy=None in 2024.12 to copy;
        # this wrapper ensures the same behaviour in older versions too.
        # https://github.com/dask/dask/pull/11524/
        assert b[0] == 1.0
    else:
        # copy=None defaults to False when possible
        assert b[0] == 0.0


@pytest.mark.parametrize("library", ["numpy", "cupy", "torch"])
def test_clip_out(library):
    """Test non-standard out= parameter for clip()

    (see "Avoid Restricting Behavior that is Outside the Scope of the Standard"
    in https://data-apis.org/array-api-compat/dev/special-considerations.html)
    """
    xp = import_(library, wrapper=True)
    x = xp.asarray([10, 20, 30])
    out = xp.zeros_like(x)
    xp.clip(x, 15, 25, out=out)
    expect = xp.asarray([15, 20, 25])
    assert xp.all(out == expect)
