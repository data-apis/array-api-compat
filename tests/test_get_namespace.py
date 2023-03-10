import array_api_compat
from array_api_compat import get_namespace
import pytest


@pytest.mark.parametrize("library", ["cupy", "numpy", "torch"])
@pytest.mark.parametrize("api_version", [None, '2021.12'])
def test_get_namespace(library, api_version):
    lib = pytest.importorskip(library)

    array = lib.asarray([1.0, 2.0, 3.0])
    namespace = array_api_compat.get_namespace(array, api_version=api_version)

    if 'array_api' in library:
        assert namespace == lib
    else:
        assert namespace == getattr(array_api_compat, library)

def test_get_namespace_multiple():
    import numpy as np

    x = np.asarray([1, 2])
    assert get_namespace(x, x) == get_namespace((x, x)) == \
        get_namespace((x, x), x) == array_api_compat.numpy

def test_get_namespace_errors():
    pytest.raises(TypeError, lambda: get_namespace([1]))
    pytest.raises(TypeError, lambda: get_namespace())

    import numpy as np
    import torch
    x = np.asarray([1, 2])
    y = torch.asarray([1, 2])

    pytest.raises(TypeError, lambda: get_namespace(x, y))

    pytest.raises(ValueError, lambda: get_namespace(x, api_version='2022.12'))
