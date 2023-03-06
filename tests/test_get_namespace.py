import array_api_compat
import pytest


@pytest.mark.parametrize("library", ["cupy", "numpy", "torch"])
def test_get_namespace(library):
    lib = pytest.importorskip(library)

    array = lib.asarray([1.0, 2.0, 3.0])
    namespace = array_api_compat.get_namespace(array)

    expected_namespace = getattr(array_api_compat, library)
    assert namespace is expected_namespace


@pytest.mark.parametrize("array_namespace", ["cupy.array_api", "numpy.array_api"])
def test_get_namespace_returns_actual_namespace(array_namespace):
    xp = pytest.importorskip(array_namespace)
    X = xp.asarray([1, 2, 3])
    xp_ = array_api_compat.get_namespace(X)
    assert xp_ is xp
