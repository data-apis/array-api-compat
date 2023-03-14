from importlib import import_module

import pytest

def import_(library):
    if library == 'cupy':
        return pytest.importorskip(library)
    return import_module(library)
