import numpy as np
import pytest
from numpy.testing import assert_allclose

from array_api_compat import to_device

from ._helpers import import_or_skip_cupy


@pytest.mark.parametrize("library", ["cupy", "numpy", "torch", "dask.array"])
def test_to_device_host(library):
    # different libraries have different semantics
    # for DtoH transfers; ensure that we support a portable
    # shim for common array libs
    # see: https://github.com/scipy/scipy/issues/18286#issuecomment-1527552919
    xp = import_or_skip_cupy("array_api_compat." + library)
    
    expected = np.array([1, 2, 3])
    x = xp.asarray([1, 2, 3])
    x = to_device(x, "cpu")
    # torch will return a genuine Device object, but
    # the other libs will do something different with
    # a `device(x)` query; however, what's really important
    # here is that we can test portably after calling
    # to_device(x, "cpu") to return to host
    assert_allclose(x, expected)
