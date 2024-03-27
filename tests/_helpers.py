from importlib import import_module

import pytest

wrapped_libraries = ["cupy", "torch", "dask.array"]
all_libraries = wrapped_libraries + ["numpy", "jax.numpy"]
import numpy as np
if np.__version__[0] == '1':
    wrapped_libraries.append("numpy")

def import_(library, wrapper=False):
    if library == 'cupy':
        pytest.importorskip(library)
    if wrapper:
        if 'jax' in library:
            library = 'jax.experimental.array_api'
        else:
            library = 'array_api_compat.' + library

    return import_module(library)
