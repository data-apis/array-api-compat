from importlib import import_module

import pytest

wrapped_libraries = ["numpy", "cupy", "torch", "dask.array"]
all_libraries = wrapped_libraries + ["jax.numpy"]

def import_(library, wrapper=False):
    if library == 'cupy':
        pytest.importorskip(library)
    if wrapper:
        if 'jax' in library:
            library = 'jax.experimental.array_api'
        else:
            library = 'array_api_compat.' + library

    return import_module(library)
