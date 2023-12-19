# Basic test that vendoring works

from .vendored._compat import dask as dask_compat

import dask.array as da
import numpy as np

def _test_numpy():
    a = dask_compat.asarray([1., 2., 3.])
    b = dask_compat.arange(3, dtype=dask_compat.float32)

    # np.pow does not exist. Update this to use something else if it is added
    res = dask_compat.pow(a, b)
    assert res.dtype == dask_compat.float64 == np.float64
    assert isinstance(a, da.array)
    assert isinstance(b, da.array)
    assert isinstance(res, da.array)

    np.testing.assert_allclose(res, [1., 2., 9.])
