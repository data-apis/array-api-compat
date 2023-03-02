import array_api_compat
import pytest


@pytest.mark.parametrize("library", ["cupy", "numpy", "torch"])
def test_get_namespace(library):
    lib = pytest.importorskip(library)

    array = lib.asarray([1.0, 2.0, 3.0])
    namespace = array_api_compat.get_namespace(array)

    expected_namespace = getattr(array_api_compat, library)
    assert namespace is expected_namespace

