import pytest


def test_vendoring_numpy():
    from vendor_test import uses_numpy

    uses_numpy._test_numpy()


def test_vendoring_cupy():
    pytest.importorskip("cupy")

    from vendor_test import uses_cupy

    uses_cupy._test_cupy()


def test_vendoring_torch():
    pytest.importorskip("torch")
    from vendor_test import uses_torch

    uses_torch._test_torch()


def test_vendoring_dask():
    pytest.importorskip("dask")
    from vendor_test import uses_dask
    uses_dask._test_dask()
