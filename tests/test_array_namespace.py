import subprocess
import sys
import warnings

import numpy as np
import pytest
import torch

import array_api_compat
from array_api_compat import array_namespace

from ._helpers import import_

@pytest.mark.parametrize("library", ["cupy", "numpy", "torch", "dask.array", "jax.numpy"])
@pytest.mark.parametrize("api_version", [None, "2021.12"])
def test_array_namespace(library, api_version):
    xp = import_(library)

    array = xp.asarray([1.0, 2.0, 3.0])
    namespace = array_api_compat.array_namespace(array, api_version=api_version)

    if "array_api" in library:
        assert namespace == xp
    else:
        if library == "dask.array":
            assert namespace == array_api_compat.dask.array
        elif library == "jax.numpy":
            import jax.experimental.array_api
            assert namespace == jax.experimental.array_api
        else:
            assert namespace == getattr(array_api_compat, library)

    # Check that array_namespace works even if jax.experimental.array_api
    # hasn't been imported yet (it monkeypatches __array_namespace__
    # onto JAX arrays, but we should support them regardless). The only way to
    # do this is to use a subprocess, since we cannot un-import it and another
    # test probably already imported it.
    if library == "jax.numpy" and sys.version_info >= (3, 9):
        code = f"""\
import sys
import jax.numpy
import array_api_compat
array = jax.numpy.asarray([1.0, 2.0, 3.0])

assert 'jax.experimental.array_api' not in sys.modules
namespace = array_api_compat.array_namespace(array, api_version={api_version!r})

import jax.experimental.array_api
assert namespace == jax.experimental.array_api
"""
        subprocess.run([sys.executable, "-c", code], check=True)

def test_array_namespace_errors():
    pytest.raises(TypeError, lambda: array_namespace([1]))
    pytest.raises(TypeError, lambda: array_namespace())

    x = np.asarray([1, 2])
    pytest.raises(TypeError, lambda: array_namespace((x, x)))
    pytest.raises(TypeError, lambda: array_namespace(x, (x, x)))

def test_array_namespace_errors_torch():
    y = torch.asarray([1, 2])
    x = np.asarray([1, 2])
    pytest.raises(TypeError, lambda: array_namespace(x, y))

def test_api_version():
    x = np.asarray([1, 2])
    np_ = import_("numpy", wrapper=True)
    assert array_namespace(x, api_version="2022.12") == np_
    assert array_namespace(x, api_version=None) == np_
    assert array_namespace(x) == np_
    # Should issue a warning
    with warnings.catch_warnings(record=True) as w:
        assert array_namespace(x, api_version="2021.12") == np_
        assert len(w) == 1
        assert "2021.12" in str(w[0].message)

    pytest.raises(ValueError, lambda: array_namespace(x, api_version="2020.12"))

def test_get_namespace():
    # Backwards compatible wrapper
    assert array_api_compat.get_namespace is array_api_compat.array_namespace
