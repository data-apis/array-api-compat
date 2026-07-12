"""MLX-specific integration tests for array-api-compat."""
import pytest

try:
    import mlx.core as mx
except ImportError:
    pytestmark = pytest.skip(allow_module_level=True, reason="mlx not found")

from array_api_compat import (
    array_namespace,
    is_mlx_array,
    is_mlx_namespace,
    device,
    to_device,
)
import array_api_compat.mlx as mlx_compat


# --- Detection helpers ---

def test_is_mlx_array():
    x = mx.array([1.0, 2.0])
    assert is_mlx_array(x)
    assert not is_mlx_array([1.0, 2.0])
    assert not is_mlx_array(1.0)


def test_is_mlx_namespace():
    assert is_mlx_namespace(mlx_compat)
    assert is_mlx_namespace(mx)
    import numpy as np
    assert not is_mlx_namespace(np)


def test_array_namespace():
    x = mx.array([1.0, 2.0])
    xp = array_namespace(x)
    assert is_mlx_namespace(xp)


# --- API version ---

def test_array_api_version():
    assert mlx_compat.__array_api_version__ == "2025.12"


# --- Renamed functions ---

def test_trig_renames():
    import math
    x = mx.array([0.5])
    assert float(mlx_compat.acos(x)[0]) == pytest.approx(math.acos(0.5))
    assert float(mlx_compat.asin(x)[0]) == pytest.approx(math.asin(0.5))
    assert float(mlx_compat.atan(x)[0]) == pytest.approx(math.atan(0.5))


def test_concat():
    a = mx.array([1, 2])
    b = mx.array([3, 4])
    result = mlx_compat.concat([a, b])
    assert list(result.tolist()) == [1, 2, 3, 4]


# --- Creation functions ---

def test_arange():
    x = mlx_compat.arange(5)
    assert list(x.tolist()) == [0, 1, 2, 3, 4]


def test_linspace():
    x = mlx_compat.linspace(0.0, 1.0, 5)
    assert len(x.tolist()) == 5


def test_zeros_ones_empty():
    assert mlx_compat.zeros((2, 3)).shape == (2, 3)
    assert mlx_compat.ones((2, 3)).shape == (2, 3)
    assert mlx_compat.full((2,), 7.0).shape == (2,)


def test_eye():
    e = mlx_compat.eye(3)
    assert e.shape == (3, 3)
    assert float(e[0, 0].tolist()) == 1.0
    assert float(e[0, 1].tolist()) == 0.0


# --- asarray / astype ---

def test_asarray():
    x = mlx_compat.asarray([1.0, 2.0], dtype=mx.float32)
    assert is_mlx_array(x)
    assert x.dtype == mx.float32


def test_astype():
    x = mx.array([1.0, 2.0], dtype=mx.float32)
    y = mlx_compat.astype(x, mx.float16)
    assert y.dtype == mx.float16


# --- iinfo shim ---

def test_iinfo():
    info = mlx_compat.iinfo(mx.int32)
    assert info.min == -(2**31)
    assert info.max == 2**31 - 1
    assert info.bits == 32

    info8 = mlx_compat.iinfo(mx.uint8)
    assert info8.min == 0
    assert info8.max == 255


def test_iinfo_from_array():
    x = mx.array([1], dtype=mx.int16)
    info = mlx_compat.iinfo(x)
    assert info.bits == 16


# --- isdtype ---

def test_isdtype():
    assert mlx_compat.isdtype(mx.float32, "real floating")
    assert mlx_compat.isdtype(mx.int32, "signed integer")
    assert mlx_compat.isdtype(mx.uint8, "unsigned integer")
    assert mlx_compat.isdtype(mx.bool_, "bool")
    assert not mlx_compat.isdtype(mx.float32, "integral")


# --- std / var (correction kwarg) ---

def test_std_var():
    x = mx.array([1.0, 2.0, 3.0, 4.0])
    s = mlx_compat.std(x, correction=1.0)
    v = mlx_compat.var(x, correction=1.0)
    assert float(s.tolist()) == pytest.approx(float(mx.sqrt(mx.array(5.0 / 3.0)).tolist()), rel=1e-5)
    _ = v  # just confirm it runs


# --- sort / argsort with descending ---

def test_sort_descending():
    x = mx.array([3, 1, 2])
    s = mlx_compat.sort(x, descending=True)
    assert list(s.tolist()) == [3, 2, 1]


def test_argsort_descending():
    x = mx.array([3, 1, 2])
    idx = mlx_compat.argsort(x, descending=True)
    assert list(idx.tolist()) == [0, 2, 1]


# --- reshape ---

def test_reshape():
    x = mx.array([1, 2, 3, 4])
    y = mlx_compat.reshape(x, (2, 2))
    assert y.shape == (2, 2)


# --- permute_dims ---

def test_permute_dims():
    x = mx.zeros((2, 3, 4))
    y = mlx_compat.permute_dims(x, (2, 0, 1))
    assert y.shape == (4, 2, 3)


# --- unstack ---

def test_unstack():
    x = mx.array([[1, 2], [3, 4]])
    parts = mlx_compat.unstack(x, axis=0)
    assert len(parts) == 2
    assert list(parts[0].tolist()) == [1, 2]


# --- cumulative_sum ---

def test_cumulative_sum():
    x = mx.array([1, 2, 3])
    cs = mlx_compat.cumulative_sum(x, axis=0)
    assert list(cs.tolist()) == [1, 3, 6]


def test_cumulative_sum_include_initial():
    x = mx.array([1, 2, 3])
    cs = mlx_compat.cumulative_sum(x, axis=0, include_initial=True)
    assert list(cs.tolist()) == [0, 1, 3, 6]


# --- __array_namespace_info__ ---

def test_namespace_info_capabilities():
    info = mlx_compat.__array_namespace_info__()
    caps = info.capabilities()
    assert caps["boolean indexing"] is True
    assert "data-dependent shapes" in caps


def test_boolean_indexing_getitem():
    x = mx.array([1, 2, 3])
    mask = x < 3
    y = x[mask]
    assert list(y.tolist()) == [1, 2]

def test_namespace_info_dtypes():
    info = mlx_compat.__array_namespace_info__()
    dtypes = info.dtypes()
    assert "float32" in dtypes
    assert "int32" in dtypes
    assert "bool" in dtypes
    # MLX has no float64
    assert "float64" not in dtypes


def test_namespace_info_default_dtypes():
    info = mlx_compat.__array_namespace_info__()
    dd = info.default_dtypes()
    assert dd["real floating"] == mx.float32


def test_namespace_info_devices():
    info = mlx_compat.__array_namespace_info__()
    devs = info.devices()
    assert len(devs) >= 1


# --- device / to_device ---

def test_device():
    x = mx.array([1.0])
    d = device(x)
    assert d is not None


def test_to_device():
    x = mx.array([1.0])
    d = device(x)
    y = to_device(x, d)
    assert is_mlx_array(y)
    assert device(y) == d


# --- linalg submodule ---

def test_linalg_available():
    assert hasattr(mlx_compat, "linalg")


def test_linalg_eigh():
    A = mx.array([[2.0, 1.0], [1.0, 2.0]])
    result = mlx_compat.linalg.eigh(A)
    assert hasattr(result, "eigenvalues")
    assert hasattr(result, "eigenvectors")


def test_linalg_eig_not_implemented():
    A = mx.array([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(NotImplementedError):
        mlx_compat.linalg.eig(A)


# --- fft submodule ---

def test_fft_available():
    assert hasattr(mlx_compat, "fft")


def test_fft_basic():
    x = mx.array([1.0, 0.0, 0.0, 0.0])
    result = mlx_compat.fft.fft(x)
    assert result.dtype == mx.complex64
