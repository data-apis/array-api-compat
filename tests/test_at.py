from __future__ import annotations

from contextlib import contextmanager, suppress

import numpy as np
import pytest

from array_api_compat import (
    array_namespace,
    at,
    is_cupy_array,
    is_dask_array,
    is_jax_array,
    is_torch_array,
    is_torch_namespace,
    is_pydata_sparse_array,
    is_writeable_array,
)
from ._helpers import import_, all_libraries


def assert_array_equal(a, b):
    if is_pydata_sparse_array(a):
        a = a.todense()
    elif is_cupy_array(a):
        a = a.get()
    elif is_dask_array(a):
        a = a.compute()
    np.testing.assert_array_equal(a, b)


@contextmanager
def assert_copy(x, copy: bool | None):
    # dask arrays are writeable, but writing to them will hot-swap the
    # dask graph inside the collection so that anything that references
    # the original graph, i.e. the input collection, won't be mutated.
    if copy is False and (not is_writeable_array(x) or is_dask_array(x)):
        with pytest.raises((TypeError, ValueError)):
            yield
        return

    xp = array_namespace(x)
    x_orig = xp.asarray(x, copy=True)
    yield

    expect_copy = not is_writeable_array(x) if copy is None else copy
    assert_array_equal((x == x_orig).all(), expect_copy)


@pytest.fixture(params=all_libraries + ["np_readonly"])
def x(request):
    library = request.param
    if library == "np_readonly":
        x = np.asarray([10, 20, 30])
        x.flags.writeable = False
    else:
        lib = import_(library)
        x = lib.asarray([10, 20, 30])
    return x


@pytest.mark.parametrize("copy", [True, False, None])
@pytest.mark.parametrize(
    "op,arg,expect",
    [
        ("set", 40, [10, 40, 40]),
        ("add", 40, [10, 60, 70]),
        ("subtract", 100, [10, -80, -70]),
        ("multiply", 2, [10, 40, 60]),
        ("divide", 2, [10, 10, 15]),
        ("power", 2, [10, 400, 900]),
        ("min", 25, [10, 20, 25]),
        ("max", 25, [10, 25, 30]),
    ],
)
def test_update_ops(x, copy, op, arg, expect):
    if is_pydata_sparse_array(x):
        pytest.skip("at() does not support updates on sparse arrays")

    with assert_copy(x, copy):
        y = getattr(at(x, slice(1, None)), op)(arg, copy=copy)
        assert isinstance(y, type(x))
        assert_array_equal(y, expect)


@pytest.mark.parametrize("copy", [True, False, None])
def test_get(x, copy):

    expect_copy = copy
    if is_dask_array(x) and copy is None:
        # dask is mutable, but __getitem__ never returns a view
        expect_copy = True

    with assert_copy(x, expect_copy):
        y = at(x, slice(2)).get(copy=copy)
        assert isinstance(y, type(x))
        assert_array_equal(y, [10, 20])
        # Let assert_copy test that y is a view or copy
        with suppress((TypeError, ValueError)):
            y[0] = 40


@pytest.mark.parametrize(
    "idx",
    [
        [0, 1],
        (0, 1),
        np.array([0, 1], dtype="int32"),
        np.array([0, 1], dtype="uint32"),
        # torch only supports tensors of native integers as indices
        lambda xp: xp.asarray([0, 1], dtype=None if is_torch_namespace(xp) else "int32"),
        lambda xp: xp.asarray([0, 1], dtype=None if is_torch_namespace(xp) else "uint32"),
        [True, True, False],
        (True, True, False),
        np.array([True, True, False]),
        lambda xp: xp.asarray([True, True, False]),
    ],
)
@pytest.mark.parametrize("tuple_index", [True, False])
def test_get_fancy_indices(x, idx, tuple_index):
    """get() with a fancy index always returns a copy"""
    if callable(idx):
        xp = array_namespace(x)
        idx = idx(xp)

    # FIXME this is unhealthy.
    # https://github.com/data-apis/array-api/issues/864
    if is_jax_array(x) and isinstance(idx, (list, tuple)):
        pytest.skip("JAX fancy indices must always be arrays")
    if is_pydata_sparse_array(x) and is_pydata_sparse_array(idx):
        pytest.skip("sparse fancy indices can't be sparse themselves")
    if is_torch_array(x) and isinstance(idx, np.ndarray) and idx.dtype.kind == "u":
        pytest.skip("torch does not support unsigned integer fancy indices")
    if is_dask_array(x) and isinstance(idx, tuple):
        pytest.skip("dask does not support tuples; only lists or arrays")
    if isinstance(idx, tuple) and not tuple_index:
        pytest.skip("tuple indices must always be wrapped in a tuple")

    if tuple_index:
        idx = (idx,)

    with assert_copy(x, True):
        y = at(x, idx).get()
        assert isinstance(y, type(x))
        assert_array_equal(y, [10, 20])
        # Let assert_copy test that y is a view or copy
        with suppress((TypeError, ValueError)):
            y[0] = 40

    with assert_copy(x, True):
        y = at(x, idx).get(copy=None)
        assert isinstance(y, type(x))
        assert_array_equal(y, [10, 20])
        # Let assert_copy test that y is a view or copy
        with suppress((TypeError, ValueError)):
            y[0] = 40

    with pytest.raises(TypeError, match="fancy index"):
        at(x, idx).get(copy=False)


def test_variant_index_syntax(x):
    y = at(x)[:2].get()
    assert isinstance(y, type(x))
    assert_array_equal(y, [10, 20])

    with pytest.raises(ValueError):
        at(x, 1)[2]
    with pytest.raises(ValueError):
        at(x)[1][2]
