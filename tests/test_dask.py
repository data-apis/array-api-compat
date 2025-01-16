from contextlib import contextmanager

import dask
import numpy as np
import pytest
import dask.array as da

from array_api_compat import array_namespace


@pytest.fixture
def xp():
    """Fixture returning the wrapped dask namespace"""
    return array_namespace(da.empty(0))


@contextmanager
def assert_no_compute():
    """
    Context manager that raises if at any point inside it anything calls compute()
    or persist(), e.g. as it can be triggered implicitly by __bool__, __array__, etc.
    """
    def get(dsk, *args, **kwargs):
        raise AssertionError("Called compute() or persist()")
    
    with dask.config.set(scheduler=get):
        yield


def test_assert_no_compute():
    """Test the assert_no_compute context manager"""
    a = da.asarray(True)
    with pytest.raises(AssertionError, match="Called compute"):
        with assert_no_compute():
            bool(a)

    # Exiting the context manager restores the original scheduler
    assert bool(a) is True


# Test no_compute for functions that use generic _aliases with xp=np

def test_unary_ops_no_compute(xp):
    with assert_no_compute():
        a = xp.asarray([1.5, -1.5])
        xp.ceil(a)
        xp.floor(a)
        xp.trunc(a)
        xp.sign(a)


def test_matmul_tensordot_no_compute(xp):
    A = da.ones((4, 4), chunks=2)
    B = da.zeros((4, 4), chunks=2)
    with assert_no_compute():
        xp.matmul(A, B)
        xp.tensordot(A, B)


# Test no_compute for functions that are fully bespoke for dask

def test_asarray_no_compute(xp):
    with assert_no_compute():
        a = xp.arange(10)
        xp.asarray(a)
        xp.asarray(a, dtype=np.int16)
        xp.asarray(a, dtype=a.dtype)
        xp.asarray(a, copy=True)
        xp.asarray(a, copy=True, dtype=np.int16)
        xp.asarray(a, copy=True, dtype=a.dtype)
        xp.asarray(a, copy=False)
        xp.asarray(a, copy=False, dtype=a.dtype)


@pytest.mark.parametrize("copy", [True, False])
def test_astype_no_compute(xp, copy):
    with assert_no_compute():
        a = xp.arange(10)
        xp.astype(a, np.int16, copy=copy)
        xp.astype(a, a.dtype, copy=copy)


def test_clip_no_compute(xp):
    with assert_no_compute():
        a = xp.arange(10)
        xp.clip(a)
        xp.clip(a, 1)
        xp.clip(a, 1, 8)


def test_generators_are_lazy(xp):
    """
    Test that generator functions are fully lazy, e.g. that
    da.ones(n) is not implemented as da.asarray(np.ones(n))
    """
    size = 100_000_000_000  # 800 GB
    chunks = size // 10  # 10x 80 GB chunks

    with assert_no_compute():
        xp.zeros(size, chunks=chunks)
        xp.ones(size, chunks=chunks)
        xp.empty(size, chunks=chunks)
        xp.full(size, fill_value=123, chunks=chunks)
        a = xp.arange(size, chunks=chunks)
        xp.zeros_like(a)
        xp.ones_like(a)
        xp.empty_like(a)
        xp.full_like(a, fill_value=123)
