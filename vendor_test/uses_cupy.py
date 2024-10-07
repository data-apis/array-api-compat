# Basic test that vendoring works

from .vendored._compat import (
    cupy as cp_compat,
    is_cupy_array,
    is_cupy_namespace,
)

import cupy as cp

def _test_cupy():
    a = cp_compat.asarray([1., 2., 3.])
    b = cp_compat.arange(3, dtype=cp_compat.float32)

    # cp.pow does not exist. Update this to use something else if it is added
    res = cp_compat.pow(a, b)
    assert res.dtype == cp_compat.float64 == cp.float64
    assert isinstance(a, cp.ndarray)
    assert isinstance(b, cp.ndarray)
    assert isinstance(res, cp.ndarray)

    cp.testing.assert_allclose(res, [1., 2., 9.])

    assert is_cupy_array(res)
    assert is_cupy_namespace(cp) and is_cupy_namespace(cp_compat)
