import subprocess
import sys
import warnings

import numpy as np
import pytest

import array_api_compat
from array_api_compat import array_namespace

from ._helpers import import_, all_libraries, wrapped_libraries

@pytest.mark.parametrize("use_compat", [True, False, None])
@pytest.mark.parametrize("api_version", [None, "2021.12", "2022.12", "2023.12"])
@pytest.mark.parametrize("library", all_libraries)
def test_array_namespace(library, api_version, use_compat):
    xp = import_(library)

    array = xp.asarray([1.0, 2.0, 3.0])
    if use_compat and library not in wrapped_libraries:
        pytest.raises(ValueError, lambda: array_namespace(array, use_compat=use_compat))
        return
    if library == "ndonnx" and api_version in ("2021.12", "2022.12"):
        pytest.skip("Unsupported API version")

    namespace = array_namespace(array, api_version=api_version, use_compat=use_compat)

    if use_compat is False or use_compat is None and library not in wrapped_libraries:
        if library == "jax.numpy" and use_compat is None:
            import jax.numpy
            if hasattr(jax.numpy, "__array_api_version__"):
                # JAX v0.4.32 or later uses jax.numpy directly
                assert namespace == jax.numpy
            else:
                # JAX v0.4.31 or earlier uses jax.experimental.array_api
                import jax.experimental.array_api
                assert namespace == jax.experimental.array_api
        else:
            assert namespace == xp
    else:
        if library == "dask.array":
            assert namespace == array_api_compat.dask.array
        else:
            assert namespace == getattr(array_api_compat, library)

    if library == "numpy":
        # check that the same namespace is returned for NumPy scalars
        scalar_namespace = array_namespace(
            xp.float64(0.0), api_version=api_version, use_compat=use_compat
        )
        assert scalar_namespace == namespace

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

if hasattr(jax.numpy, '__array_api_version__'):
    assert namespace == jax.numpy
else:
    import jax.experimental.array_api
    assert namespace == jax.experimental.array_api
"""
        subprocess.run([sys.executable, "-c", code], check=True)

def test_jax_zero_gradient():
    jax = import_("jax")
    jx = jax.numpy.arange(4)
    jax_zero = jax.vmap(jax.grad(jax.numpy.float32, allow_int=True))(jx)
    assert array_namespace(jax_zero) is array_namespace(jx)

def test_array_namespace_errors():
    pytest.raises(TypeError, lambda: array_namespace([1]))
    pytest.raises(TypeError, lambda: array_namespace())

    x = np.asarray([1, 2])
    pytest.raises(TypeError, lambda: array_namespace((x, x)))
    pytest.raises(TypeError, lambda: array_namespace(x, (x, x)))

def test_array_namespace_errors_torch():
    torch = import_("torch")
    y = torch.asarray([1, 2])
    x = np.asarray([1, 2])
    pytest.raises(TypeError, lambda: array_namespace(x, y))

def test_api_version_torch():
    torch = import_("torch")
    x = torch.asarray([1, 2])
    torch_ = import_("torch", wrapper=True)
    assert array_namespace(x, api_version="2023.12") == torch_
    assert array_namespace(x, api_version=None) == torch_
    assert array_namespace(x) == torch_
    # Should issue a warning
    with warnings.catch_warnings(record=True) as w:
        assert array_namespace(x, api_version="2021.12") == torch_
    assert len(w) == 1
    assert "2021.12" in str(w[0].message)

    # Should issue a warning
    with warnings.catch_warnings(record=True) as w:
        assert array_namespace(x, api_version="2022.12") == torch_
    assert len(w) == 1
    assert "2022.12" in str(w[0].message)

    pytest.raises(ValueError, lambda: array_namespace(x, api_version="2020.12"))

def test_get_namespace():
    # Backwards compatible wrapper
    assert array_api_compat.get_namespace is array_namespace

def test_python_scalars():
    torch = import_("torch")
    a = torch.asarray([1, 2])
    xp = import_("torch", wrapper=True)

    pytest.raises(TypeError, lambda: array_namespace(1))
    pytest.raises(TypeError, lambda: array_namespace(1.0))
    pytest.raises(TypeError, lambda: array_namespace(1j))
    pytest.raises(TypeError, lambda: array_namespace(True))
    pytest.raises(TypeError, lambda: array_namespace(None))

    assert array_namespace(a, 1) == xp
    assert array_namespace(a, 1.0) == xp
    assert array_namespace(a, 1j) == xp
    assert array_namespace(a, True) == xp
    assert array_namespace(a, None) == xp
