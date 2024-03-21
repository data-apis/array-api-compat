from array_api_compat import (is_numpy_array, is_cupy_array, is_torch_array, # noqa: F401
                              is_dask_array, is_jax_array)

from array_api_compat import is_array_api_obj, device, to_device

from ._helpers import import_

import pytest
import numpy as np
from numpy.testing import assert_allclose

is_functions = {
    'numpy': 'is_numpy_array',
    'cupy': 'is_cupy_array',
    'torch': 'is_torch_array',
    'dask.array': 'is_dask_array',
    'jax.numpy': 'is_jax_array',
}

@pytest.mark.parametrize('library', is_functions.keys())
@pytest.mark.parametrize('func', is_functions.values())
def test_is_xp_array(library, func):
    lib = import_(library)
    is_func = globals()[func]

    x = lib.asarray([1, 2, 3])

    assert is_func(x) == (func == is_functions[library])

    assert is_array_api_obj(x)

@pytest.mark.parametrize("library", ["cupy", "numpy", "torch", "dask.array", "jax.numpy"])
def test_device(library):
    xp = import_(library, wrapper=True)

    # We can't test much for device() and to_device() other than that
    # x.to_device(x.device) works.

    x = xp.asarray([1, 2, 3])
    dev = device(x)

    x2 = to_device(x, dev)
    assert device(x) == device(x2)


@pytest.mark.parametrize("library", ["cupy", "numpy", "torch", "dask.array"])
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


@pytest.mark.parametrize("target_library", is_functions.keys())
@pytest.mark.parametrize("source_library", is_functions.keys())
def test_asarray(source_library, target_library, request):
    if source_library == "dask.array" and target_library == "torch":
        # Allow rest of test to execute instead of immediately xfailing
        # xref https://github.com/pandas-dev/pandas/issues/38902

        # TODO: remove xfail once
        # https://github.com/dask/dask/issues/8260 is resolved
        request.node.add_marker(pytest.mark.xfail(reason="Bug in dask raising error on conversion"))
    if source_library == "cupy" and target_library != "cupy":
        # cupy explicitly disallows implicit conversions to CPU
        pytest.skip(reason="cupy does not support implicit conversion to CPU")
    src_lib = import_(source_library, wrapper=True)
    tgt_lib = import_(target_library, wrapper=True)
    is_tgt_type = globals()[is_functions[target_library]]

    a = src_lib.asarray([1, 2, 3])
    b = tgt_lib.asarray(a)

    assert is_tgt_type(b), f"Expected {b} to be a {tgt_lib.ndarray}, but was {type(b)}"
