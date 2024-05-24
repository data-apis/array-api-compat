"""
Test that array_api_compat has no "hard" dependencies.

Libraries like NumPy should only be imported if a numpy array is passed to
array_namespace or if array_api_compat.numpy is explicitly imported.

We have to test this in a subprocess because these libraries have already been
imported from the other tests.
"""

import sys
import subprocess

import pytest

class Array:
    # Dummy array namespace that doesn't depend on any array library
    def __array_namespace__(self, api_version=None):
        class Namespace:
            pass
        return Namespace()

def _test_dependency(mod):
    assert mod not in sys.modules

    # Run various functions that shouldn't depend on mod and check that they
    # don't import it.

    import array_api_compat
    assert mod not in sys.modules

    a = Array()

    # array-api-strict is an example of an array API library that isn't
    # wrapped by array-api-compat.
    if "strict" not in mod and mod != "sparse":
        is_mod_array = getattr(array_api_compat, f"is_{mod.split('.')[0]}_array")
        assert not is_mod_array(a)
        assert mod not in sys.modules

    is_array_api_obj = getattr(array_api_compat, "is_array_api_obj")
    assert is_array_api_obj(a)
    assert mod not in sys.modules

    array_namespace = getattr(array_api_compat, "array_namespace")
    array_namespace(Array())
    assert mod not in sys.modules

# TODO: Test that wrapper for library X doesn't depend on wrappers for library
# Y (except most array libraries actually do themselves depend on numpy).

@pytest.mark.parametrize("library", ["cupy", "numpy", "torch", "dask.array",
                                     "jax.numpy", "sparse", "array_api_strict"])
def test_numpy_dependency(library):
    # This import is here because it imports numpy
    from ._helpers import import_

    # This unfortunately won't go through any of the pytest machinery. We
    # reraise the exception as an AssertionError so that pytest will show it
    # in a semi-reasonable way

    # Import (in this process) to make sure 'library' is actually installed and
    # so that cupy can be skipped.
    import_(library)

    try:
        subprocess.run([sys.executable, '-c', f'''\
from tests.test_no_dependencies import _test_dependency

_test_dependency({library!r})'''], check=True, capture_output=True, encoding='utf-8')
    except subprocess.CalledProcessError as e:
        print(e.stdout, end='')
        raise AssertionError(e.stderr)
