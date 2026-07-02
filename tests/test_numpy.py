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
    np.testing.assert_array_equal(out, xp.asarray([[15, 20, 30], [25, 45, 60]], dtype=np.uint8))


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
