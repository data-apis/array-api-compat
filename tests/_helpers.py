from importlib import import_module
import sys

import pytest

wrapped_libraries = ["cupy", "torch", "dask.array"]
all_libraries = wrapped_libraries + ["numpy", "jax.numpy", "sparse"]
import numpy as np
if np.__version__[0] == '1':
    wrapped_libraries.append("numpy")

# `sparse` added array API support as of Python 3.10.
if sys.version_info >= (3, 10):
    all_libraries.append('sparse')

def import_(library, wrapper=False):
    # CuPy requires a GPU
    # `sparse` has a dependency conflict with NumPy 1.21
    if library in {'cupy', 'sparse'}:
        pytest.importorskip(library)
    if wrapper:
        if 'jax' in library:
            library = 'jax.experimental.array_api'
        elif library.startswith('sparse'):
            library = 'sparse'
        else:
            library = 'array_api_compat.' + library

    return import_module(library)
