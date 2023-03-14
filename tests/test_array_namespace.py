import array_api_compat
from array_api_compat import array_namespace

from ._helpers import import_

import pytest


@pytest.mark.parametrize("library", ["cupy", "numpy", "torch"])
@pytest.mark.parametrize("api_version", [None, '2021.12'])
def test_array_namespace(library, api_version):
    lib = import_(library)

    array = lib.asarray([1.0, 2.0, 3.0])
    namespace = array_api_compat.array_namespace(array, api_version=api_version)

    if 'array_api' in library:
        assert namespace == lib
    else:
        assert namespace == getattr(array_api_compat, library)

def test_array_namespace_multiple():
    import numpy as np

    x = np.asarray([1, 2])
    assert array_namespace(x, x) == array_namespace((x, x)) == \
        array_namespace((x, x), x) == array_api_compat.numpy

def test_array_namespace_errors():
    pytest.raises(TypeError, lambda: array_namespace([1]))
    pytest.raises(TypeError, lambda: array_namespace())

    import numpy as np
    import torch
    x = np.asarray([1, 2])
    y = torch.asarray([1, 2])

    pytest.raises(TypeError, lambda: array_namespace(x, y))

    pytest.raises(ValueError, lambda: array_namespace(x, api_version='2022.12'))

def test_get_namespace():
    # Backwards compatible wrapper
    assert array_api_compat.get_namespace is array_api_compat.array_namespace
