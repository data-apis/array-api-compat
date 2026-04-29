"""Test "unspecified" behavior which we cannot easily test in the Array API test suite.
"""
import warnings
import pytest

try:
    import numpy as np
except ImportError:
    pytestmark = pytest.skip(allow_module_level=True, reason="numpy not found")

from array_api_compat import is_array_api_obj

def test_matrix_is_not_array_api_obj():
    assert is_array_api_obj(np.asarray(3))
    assert is_array_api_obj(np.float64(3))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        assert not is_array_api_obj(np.matrix(3))
