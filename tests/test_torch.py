"""Test "unspecified" behavior which we cannot easily test in the Array API test suite.
"""
import itertools

import pytest

try:
    import torch
except ImportError:
    pytestmark = pytest.skip(allow_module_level=True, reason="pytorch not found")

from array_api_compat import torch as xp


class TestResultType:
    def test_empty(self):
        with pytest.raises(ValueError):
            xp.result_type()

    def test_one_arg(self):
        for x in [1, 1.0, 1j, '...', None]:
            with pytest.raises((ValueError, AttributeError)):
                xp.result_type(x)

        for x in [xp.float32, xp.int64, torch.complex64]:
            assert xp.result_type(x) == x

        for x in [xp.asarray(True, dtype=xp.bool), xp.asarray(1, dtype=xp.complex64)]:
            assert xp.result_type(x) == x.dtype

    def test_two_args(self):
        # Only include here things "unspecified" in the spec

        # scalar, tensor or tensor,tensor
        for x, y in [
            (1., 1j),
            (1j, xp.arange(3)),
            (True, xp.asarray(3.)),
            (xp.ones(3) == 1, 1j*xp.ones(3)),
        ]:
            assert xp.result_type(x, y) == torch.result_type(x, y)

        # dtype, scalar
        for x, y in [
            (1j, xp.int64),
            (True, xp.float64),
        ]:
            assert xp.result_type(x, y) == torch.result_type(x, xp.empty([], dtype=y))

        # dtype, dtype
        for x, y in [
            (xp.bool, xp.complex64)
        ]:
            xt, yt = xp.empty([], dtype=x), xp.empty([], dtype=y)
            assert xp.result_type(x, y) == torch.result_type(xt, yt)

    def test_multi_arg(self):
        torch.set_default_dtype(torch.float32)

        args = [1., 5, 3, torch.asarray([3], dtype=torch.float16), 5, 6, 1.]
        assert xp.result_type(*args) == torch.float16

        args = [1, 2, 3j, xp.arange(3, dtype=xp.float32), 4, 5, 6]
        assert xp.result_type(*args) == xp.complex64

        args = [1, 2, 3j, xp.float64, 4, 5, 6]
        assert xp.result_type(*args) == xp.complex128

        args = [1, 2, 3j, xp.float64, 4, xp.asarray(3, dtype=xp.int16), 5, 6, False]
        assert xp.result_type(*args) == xp.complex128

        i64 = xp.ones(1, dtype=xp.int64)
        f16 = xp.ones(1, dtype=xp.float16)
        for i in itertools.permutations([i64, f16, 1.0, 1.0]):
            assert xp.result_type(*i) == xp.float16, f"{i}"

        with pytest.raises(ValueError):
            xp.result_type(1, 2, 3, 4)


    @pytest.mark.parametrize("default_dt", ['float32', 'float64'])
    @pytest.mark.parametrize("dtype_a",
        (xp.int32, xp.int64, xp.float32, xp.float64, xp.complex64, xp.complex128)
    )
    @pytest.mark.parametrize("dtype_b", 
        (xp.int32, xp.int64, xp.float32, xp.float64, xp.complex64, xp.complex128)
    )
    def test_gh_273(self, default_dt, dtype_a, dtype_b):
        # Regression test for https://github.com/data-apis/array-api-compat/issues/273

        try:
            prev_default = torch.get_default_dtype()
            default_dtype = getattr(torch, default_dt)
            torch.set_default_dtype(default_dtype)

            a = xp.asarray([2, 1], dtype=dtype_a)
            b = xp.asarray([1, -1], dtype=dtype_b)
            dtype_1 = xp.result_type(a, b, 1.0)
            dtype_2 = xp.result_type(b, a, 1.0)
            assert dtype_1 == dtype_2
        finally:
            torch.set_default_dtype(prev_default)


def test_meshgrid():
    """Verify that array_api_compat.torch.meshgrid defaults to indexing='xy'."""

    x, y = xp.asarray([1, 2]), xp.asarray([4])

    X, Y = xp.meshgrid(x, y)

    # output of torch.meshgrid(x, y, indexing='xy') -- indexing='ij' is different
    X_xy, Y_xy = xp.asarray([[1, 2]]), xp.asarray([[4, 4]])

    assert X.shape == X_xy.shape
    assert xp.all(X == X_xy)

    assert Y.shape == Y_xy.shape
    assert xp.all(Y == Y_xy)
