"""Test "unspecified" behavior which we cannot easily test in the Array API test suite."""

import warnings
import pytest

try:
    import numpy as np
except ImportError:
    pytestmark = pytest.skip(allow_module_level=True, reason="numpy not found")

from array_api_compat import is_array_api_obj


def test_numpy_clip_out_and_broadcast():
    from array_api_compat import numpy as xp

    x = xp.asarray([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)
    min_bound = xp.asarray([15, 35, 55], dtype=np.int16)
    max_bound = xp.asarray([25, 45, 65], dtype=np.int16)
    out = xp.empty_like(x)

    result = xp.clip(x, min_bound, max_bound, out=out)

    assert result is out
    assert out.dtype == x.dtype
    np.testing.assert_array_equal(out, xp.asarray([[15, 35, 55], [25, 45, 60]], dtype=np.uint8))


def test_numpy_clip_uint8_casts_bounds_outside_range():
    from array_api_compat import numpy as xp

    x = xp.asarray([0, 10, 250], dtype=np.uint8)
    min_bound = np.int16(-1)
    max_bound = np.int16(200)

    result = xp.clip(x, min_bound, max_bound)

    assert result.dtype == x.dtype
    np.testing.assert_array_equal(result, xp.asarray([200, 200, 200], dtype=np.uint8))


def test_numpy_clip_int64_casts_bounds_outside_range():
    from array_api_compat import numpy as xp

    x = xp.asarray([-(2**63), -1, 0, 2**63 - 1], dtype=np.int64)
    min_bound = np.float64(-1e20)
    max_bound = np.float64(1e20)

    result = xp.clip(x, min_bound, max_bound)

    assert result.dtype == x.dtype
    np.testing.assert_array_equal(
        result,
        xp.asarray(
            [
                np.iinfo(np.int64).min,
                np.iinfo(np.int64).min,
                np.iinfo(np.int64).min,
                np.iinfo(np.int64).min,
            ],
            dtype=np.int64,
        ),
    )


def test_numpy_clip_float16_casts_bounds_outside_range():
    from array_api_compat import numpy as xp

    x = xp.asarray([0.0, 1.5, 3.0], dtype=np.float16)
    min_bound = np.float32(-1e10)
    max_bound = np.float32(2.0)

    result = xp.clip(x, min_bound, max_bound)

    assert result.dtype == x.dtype
    np.testing.assert_array_equal(result, xp.asarray([0.0, 1.5, 2.0], dtype=np.float16))


def test_numpy_clip_returns_copy_when_unbounded():
    from array_api_compat import numpy as xp

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
