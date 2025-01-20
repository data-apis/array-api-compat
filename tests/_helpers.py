from importlib import import_module

import pytest

wrapped_libraries = ["numpy", "cupy", "torch", "dask.array"]
all_libraries = wrapped_libraries + ["array_api_strict", "jax.numpy", "sparse"]


def import_(library, wrapper=False):
    if library == 'cupy':
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
