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
