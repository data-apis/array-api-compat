from importlib import import_module

import pytest

wrapped_libraries = ["numpy", "cupy", "torch", "dask.array"]
all_libraries = wrapped_libraries + [
    "array_api_strict", "jax.numpy", "ndonnx", "sparse"
]

def import_(library, wrapper=False):
    pytest.importorskip(library)
    if wrapper:
        if 'jax' in library:
            # JAX v0.4.32 implements the array API directly in jax.numpy
            # Older jax versions use jax.experimental.array_api
            jax_numpy = import_module("jax.numpy")
            if not hasattr(jax_numpy, "__array_api_version__"):
                library = 'jax.experimental.array_api'
        elif library in wrapped_libraries:
            library = 'array_api_compat.' + library

    return import_module(library)


def xfail(request: pytest.FixtureRequest, reason: str) -> None:
    """
    XFAIL the currently running test.

    Unlike ``pytest.xfail``, allow rest of test to execute instead of immediately
    halting it, so that it may result in a XPASS.
    xref https://github.com/pandas-dev/pandas/issues/38902
    """
    request.node.add_marker(pytest.mark.xfail(reason=reason))
