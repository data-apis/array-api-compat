import numpy as np
import pytest
import torch

import array_api_compat
from array_api_compat import array_namespace

from ._helpers import import_


@pytest.mark.parametrize("library", ["cupy", "numpy", "torch", "dask.array"])
@pytest.mark.parametrize("api_version", [None, "2021.12"])
def test_array_namespace(library, api_version):
    xp = import_(library)

    array = xp.asarray([1.0, 2.0, 3.0])
    namespace = array_api_compat.array_namespace(array, api_version=api_version)

    if "array_api" in library:
        assert namespace == xp
    else:
        if library == "dask.array":
            assert namespace == array_api_compat.dask.array
        else:
            assert namespace == getattr(array_api_compat, library)


def test_array_namespace_errors():
    pytest.raises(TypeError, lambda: array_namespace([1]))
    pytest.raises(TypeError, lambda: array_namespace())

    x = np.asarray([1, 2])
    pytest.raises(TypeError, lambda: array_namespace((x, x)))
    pytest.raises(TypeError, lambda: array_namespace(x, (x, x)))


def test_array_namespace_errors_torch():
    y = torch.asarray([1, 2])
    x = np.asarray([1, 2])
    pytest.raises(TypeError, lambda: array_namespace(x, y))
    pytest.raises(ValueError, lambda: array_namespace(x, api_version="2022.12"))


def test_get_namespace():
    # Backwards compatible wrapper
    assert array_api_compat.get_namespace is array_api_compat.array_namespace
