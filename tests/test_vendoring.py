from pytest import skip

def test_vendoring_numpy():
    from vendor_test import uses_numpy
    uses_numpy._test_numpy()


def test_vendoring_cupy():
    try:
        import cupy
    except ImportError:
        skip("CuPy is not installed")

    from vendor_test import uses_cupy
    uses_cupy._test_cupy()

def test_vendoring_torch():
    from vendor_test import uses_torch
    uses_torch._test_torch()
