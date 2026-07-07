"""Test "unspecified" behavior which we cannot easily test in the Array API test suite."""

import warnings
import pytest

try:
    import numpy as np
except ImportError:
    pytestmark = pytest.skip(allow_module_level=True, reason="numpy not found")

from array_api_compat import is_array_api_obj
from array_api_compat import numpy as xp

def test_numpy_clip_out_and_broadcast():

    x = xp.asarray([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)
    min_bound = xp.asarray([15, 35, 55], dtype=np.int16)
    max_bound = xp.asarray([25, 45, 65], dtype=np.int16)
    out = xp.empty_like(x)

    result = xp.clip(x, min_bound, max_bound, out=out)

    np.testing.assert_array_equal(result, xp.asarray([[15, 35, 55], [25, 45, 60]], dtype=np.uint8))
    assert result.dtype == x.dtype
    np.testing.assert_array_equal(out, xp.asarray([[15, 35, 55], [25, 45, 60]], dtype=np.uint8))


def test_numpy_clip_all_bounds_work_with_int_arrays():
    """Test that integer bounds outside the range of input dtype still work for integer arrays"""
    x = xp.asarray([0, 10, 250], dtype=np.uint8)
    min_bound = np.int16(-1)
    max_bound = np.int16(200)

    result = xp.clip(x, min_bound, max_bound)

    assert result.dtype == x.dtype
    np.testing.assert_array_equal(result, xp.asarray([0, 10, 200], dtype=np.uint8))


    # min and max bounds are below what can be represented by int64, 
    # so they should be clipped to the min/max of int64
    x = xp.asarray([-(2**63), -1, 0, 2**63 - 1], dtype=np.int64)
    min_bound = np.float64(-1e20)
    max_bound = np.float64(1e20)

    result = xp.clip(x, min_bound, max_bound)

    assert result.dtype == x.dtype
    np.testing.assert_array_equal(
        result,
        xp.asarray(
            [
            -(2**63), -1, 0, 2**63 - 1
            ],
            dtype=np.int64,
        ),
    )


def test_array_min_max_broadcasting_when_clipped():
    """ Tests a min as tuple list array of floats input for an integer array
    Should be clipped by min/max of the integer array"""
    x=xp.asarray([0, 10, 100], dtype=np.uint8)
    min_bound = xp.asarray([np.float64(-1e20), 5.0, 200.0], dtype=np.float32)
    max_bound = None
    result=xp.clip(x, min_bound, max_bound)
    assert result.dtype == x.dtype
    np.testing.assert_array_equal(result, xp.asarray([0, 10, 200], dtype=np.uint8))

    # now test with a tuple
    min_bound = (np.float64(-1e20), 5.0, 200.0)
    result=xp.clip(x, min_bound, max_bound)
    assert result.dtype == x.dtype
    np.testing.assert_array_equal(result, xp.asarray([0, 10, 200], dtype=np.uint8))
    
    # test with a list
    min_bound = [np.float64(-1e20), 5.0, 200.0]
    result=xp.clip(x, min_bound, max_bound)
    assert result.dtype == x.dtype
    np.testing.assert_array_equal(result, xp.asarray([0, 10, 200], dtype=np.uint8))

def test_numpy_type_promotion():
    """ Added to address comment from main alias file:
    # np.clip does type promotion but the array API clip requires that the
    # output have the same dtype as x. We do this instead of just downcasting
    # the result of xp.clip() to handle some corner cases better (e.g.,
    # avoiding uint64 -> float64 promotion).
    """
    # ensure clipping with float bounds
    x = xp.asarray([-(2**63), -1, 0, 2**63 - 1], dtype=np.int64)
    min_bound = np.float64(-1.0001)
    max_bound = np.float64(1.0001)
    result = xp.clip(x, min_bound, max_bound)
    assert result.dtype == x.dtype
    np.testing.assert_array_equal(result, xp.asarray([-1, -1, 0, 1], dtype=np.int64))
    
    # ensure clipping with int16 bounds
    min_bound = np.int16(-1)
    max_bound = np.int16(1)
    result = xp.clip(x, min_bound, max_bound)
    assert result.dtype == x.dtype
    np.testing.assert_array_equal(result, xp.asarray([-1, -1, 0, 1], dtype=np.int64))
    
    # final test with uint8 image and int64 bounds
    x = xp.asarray([0, 10, 250], dtype=np.uint8)
    min_bound = np.int64(-1)
    max_bound = np.int64(32)
    result = xp.clip(x, min_bound, max_bound)
    assert result.dtype == x.dtype
    np.testing.assert_array_equal(result, xp.asarray([0, 10, 32], dtype=np.uint8))
    
def test_numpy_clip_float16_casts_bounds_outside_range():
    """Test that float16 bounds outside the range of input dtype still work for float16 arrays"""
    x = xp.asarray([0.0, 1.5, 3.0], dtype=np.float16)
    min_bound = np.float32(-1e10) # outside of float16 range
    max_bound = np.float32(2.0)

    result = xp.clip(x, min_bound, max_bound)

    assert result.dtype == x.dtype
    np.testing.assert_array_equal(result, xp.asarray([0.0, 1.5, 2.0], dtype=np.float16))


def test_numpy_clip_returns_copy_when_unbounded():

    x = xp.arange(8, dtype=np.int64)

    y = xp.clip(x)

    assert y.dtype == x.dtype
    assert not np.shares_memory(x, y)
    np.testing.assert_array_equal(y, x)


def test_matrix_is_not_array_api_obj():
    assert is_array_api_obj(np.asarray(3))
    assert is_array_api_obj(np.float64(3))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        assert not is_array_api_obj(np.matrix(3))
