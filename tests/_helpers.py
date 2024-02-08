from importlib import import_module

import pytest


def import_or_skip_cupy(library):
    if "cupy" in library:
        return pytest.importorskip(library)
    return import_module(library)
