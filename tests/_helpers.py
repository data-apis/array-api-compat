from importlib import import_module

import sys

import pytest


def import_(library, wrapper=False):
    if 'jax' in library and sys.version_info < (3, 9):
        pytest.skip('JAX array API support does not support Python 3.8')

    if library == 'cupy':
        pytest.importorskip(library)
    if wrapper:
        if 'jax' in library:
            library = 'jax.experimental.array_api'
        else:
            library = 'array_api_compat.' + library

    return import_module(library)
