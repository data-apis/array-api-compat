from importlib import import_module

import pytest

def import_(library):
    if 'cupy' in library:
        return pytest.importorskip(library)
    return import_module(library)
