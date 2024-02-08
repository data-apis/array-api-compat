from importlib import import_module

import pytest


def import_(library, wrapper=False):
    if library == 'cupy':
        return pytest.importorskip(library)

    if wrapper:
        if 'jax' in library:
            library = 'jax.experimental.array_api'
        else:
            library = 'array_api_compat.' + library

    return import_module(library)
