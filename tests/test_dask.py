from contextlib import contextmanager

import numpy as np
import pytest

try:
    import dask
    import dask.array as da
except ImportError:
    pytestmark = pytest.skip(allow_module_level=True, reason="dask not found")

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


@pytest.mark.parametrize("chunks", (5, 10))
def test_sort_argsort_nocompute(xp, chunks):
    with assert_no_compute():
        a = xp.arange(10, chunks=chunks)
        xp.sort(a)
        xp.argsort(a)


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


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("func", ["sort", "argsort"])
def test_sort_argsort_chunks(xp, func, axis):
    """Test that sort and argsort are functionally correct when
    the array is chunked along the sort axis, e.g. the sort is
    not just local to each chunk.
    """
    a = da.random.random((10, 10), chunks=(5, 5))
    actual = getattr(xp, func)(a, axis=axis)
    expect = getattr(np, func)(a.compute(), axis=axis)
    np.testing.assert_array_equal(actual, expect)


@pytest.mark.parametrize(
    "shape,chunks",
    [
        # 3 GiB; 128 MiB per chunk; must rechunk before sorting.
        # Sort chunks can be 128 MiB each; no need for final rechunk.
        ((20_000, 20_000), "auto"),
        # 3 GiB; 128 MiB per chunk; must rechunk before sorting.
        # Must sort on two 1.5 GiB chunks; benefits from final rechunk.
        ((2, 2**30 * 3 // 16), "auto"),
        # 3 GiB; 1.5 GiB per chunk; no need to rechunk before sorting.
        # Surely the user must know what they're doing, so don't
        # perform the final rechunk.
        ((2, 2**30 * 3 // 16), (1, -1)),
    ],
)
@pytest.mark.parametrize("func", ["sort", "argsort"])
def test_sort_argsort_chunk_size(xp, func, shape, chunks):
    """
    Test that sort and argsort produce reasonably-sized chunks
    in the output array, even if they had to go through a singular
    huge one to perform the operation.
    """
    a = da.random.random(shape, chunks=chunks)
    b = getattr(xp, func)(a)
    max_chunk_size = max(b.chunks[0]) * max(b.chunks[1]) * b.dtype.itemsize
    assert (
        max_chunk_size <= 128 * 1024 * 1024  # 128 MiB
        or b.chunks == a.chunks
    )


@pytest.mark.parametrize("func", ["sort", "argsort"])
def test_sort_argsort_meta(xp, func):
    """Test meta-namespace other than numpy"""
    mxp = pytest.importorskip("array_api_strict")
    typ = type(mxp.asarray(0))
    a = da.random.random(10)
    b = a.map_blocks(mxp.asarray)
    assert isinstance(b._meta, typ)
    c = getattr(xp, func)(b)
    assert isinstance(c._meta, typ)
    d = c.compute()
    # Note: np.sort(array_api_strict.asarray(0)) would return a numpy array
    assert isinstance(d, typ)
    np.testing.assert_array_equal(d, getattr(np, func)(a.compute()))
