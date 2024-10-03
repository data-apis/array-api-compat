# Basic test that vendoring works

from .vendored._compat.dask import array as dask_compat
from .vendored._compat import is_dask_array, is_dask_namespace

import dask.array as da
import numpy as np

def _test_dask():
    a = dask_compat.asarray([1., 2., 3.])
    b = dask_compat.arange(3, dtype=dask_compat.float32)

    # np.pow does not exist. Update this to use something else if it is added
    res = dask_compat.pow(a, b)
    assert res.dtype == dask_compat.float64 == np.float64
    assert isinstance(a, da.Array)
    assert isinstance(b, da.Array)
    assert isinstance(res, da.Array)

    np.testing.assert_allclose(res, [1., 2., 9.])

    assert is_dask_array(res)
    assert is_dask_namespace(da) and is_dask_namespace(dask_compat)
