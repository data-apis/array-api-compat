import subprocess
import sys
import warnings

import numpy as np
import pytest

import array_api_compat
from array_api_compat import array_namespace

from ._helpers import all_libraries, wrapped_libraries, xfail


@pytest.mark.parametrize("use_compat", [True, False, None])
@pytest.mark.parametrize(
    "api_version", [None, "2021.12", "2022.12", "2023.12", "2024.12"]
)
@pytest.mark.parametrize("library", all_libraries)
def test_array_namespace(request, library, api_version, use_compat):
    xp = pytest.importorskip(library)

    array = xp.asarray([1.0, 2.0, 3.0])
    if use_compat and library not in wrapped_libraries:
        pytest.raises(ValueError, lambda: array_namespace(array, use_compat=use_compat))
        return
    if (library == "sparse" and api_version in ("2023.12", "2024.12")) or (
        library == "jax.numpy" and api_version in ("2021.12", "2022.12", "2023.12")
    ):
        xfail(request, "Unsupported API version")

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        namespace = array_namespace(array, api_version=api_version, use_compat=use_compat)

    if use_compat is False or use_compat is None and library not in wrapped_libraries:
        if library == "jax.numpy" and not hasattr(xp, "__array_api_version__"):
            # Backwards compatibility for JAX <0.4.32
            import jax.experimental.array_api
            assert namespace == jax.experimental.array_api
        else:
            assert namespace == xp
    elif library == "dask.array":
        assert namespace == array_api_compat.dask.array
    else:
        assert namespace == getattr(array_api_compat, library)

    if library == "numpy":
        # check that the same namespace is returned for NumPy scalars
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)

            scalar_namespace = array_namespace(
                xp.float64(0.0), api_version=api_version, use_compat=use_compat
            )
            assert scalar_namespace == namespace


def test_jax_backwards_compat():
    """On JAX <0.4.32, test that array_namespace works even if
    jax.experimental.array_api has not been imported yet.
    """
    pytest.importorskip("jax")
    code = """\
import sys
import jax.numpy
import array_api_compat

array = jax.numpy.asarray([1.0, 2.0, 3.0])
assert 'jax.experimental.array_api' not in sys.modules
namespace = array_api_compat.array_namespace(array)

if hasattr(jax.numpy, '__array_api_version__'):
    assert namespace == jax.numpy
else:
    import jax.experimental.array_api
    assert namespace == jax.experimental.array_api
"""
    subprocess.check_call([sys.executable, "-c", code])


def test_jax_zero_gradient():
    jax = pytest.importorskip("jax")
    jx = jax.numpy.arange(4)
    jax_zero = jax.vmap(jax.grad(jax.numpy.float32, allow_int=True))(jx)
    assert array_namespace(jax_zero) is array_namespace(jx)


def test_array_namespace_errors():
    pytest.raises(TypeError, lambda: array_namespace([1]))
    pytest.raises(TypeError, lambda: array_namespace())

    x = np.asarray([1, 2])
    pytest.raises(TypeError, lambda: array_namespace((x, x)))
    pytest.raises(TypeError, lambda: array_namespace(x, (x, x)))


@pytest.mark.parametrize("library", all_libraries)
def test_array_namespace_many_args(library):
    xp = pytest.importorskip(library)
    a = xp.asarray(1)
    b = xp.asarray(2)
    assert array_namespace(a, b) is array_namespace(a)


def test_array_namespace_mismatch():
    xp = pytest.importorskip("array_api_strict")
    with pytest.raises(TypeError, match="Multiple namespaces"):
        array_namespace(np.asarray(1), xp.asarray(1))


def test_get_namespace():
    # Backwards compatible wrapper
    assert array_api_compat.get_namespace is array_namespace


@pytest.mark.parametrize("library", all_libraries)
def test_python_scalars(library):
    xp = pytest.importorskip(library)
    a = xp.asarray([1, 2])
    xp = array_namespace(a)

    pytest.raises(TypeError, lambda: array_namespace(1))
    pytest.raises(TypeError, lambda: array_namespace(1.0))
    pytest.raises(TypeError, lambda: array_namespace(1j))
    pytest.raises(TypeError, lambda: array_namespace(True))
    pytest.raises(TypeError, lambda: array_namespace(None))

    assert array_namespace(a, 1) is xp
    assert array_namespace(a, 1.0) is xp
    assert array_namespace(a, 1j) is xp
    assert array_namespace(a, True) is xp
    assert array_namespace(a, None) is xp
