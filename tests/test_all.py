"""Test exported names"""

import builtins

import numpy as np
import pytest

from ._helpers import wrapped_libraries

NAMES = {
    "": [
        # Inspection
        "__array_api_version__",
        "__array_namespace_info__",
        # Constants
        "e",
        "inf",
        "nan",
        "newaxis",
        "pi",
        # Creation functions
        "arange",
        "asarray",
        "empty",
        "empty_like",
        "eye",
        "from_dlpack",
        "full",
        "full_like",
        "linspace",
        "meshgrid",
        "ones",
        "ones_like",
        "tril",
        "triu",
        "zeros",
        "zeros_like",
        # Data Type Functions
        "astype",
        "can_cast",
        "finfo",
        "iinfo",
        "isdtype",
        "result_type",
        # Data Types
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
        "complex64",
        "complex128",
        # Elementwise Functions
        "abs",
        "acos",
        "acosh",
        "add",
        "asin",
        "asinh",
        "atan",
        "atan2",
        "atanh",
        "bitwise_and",
        "bitwise_left_shift",
        "bitwise_invert",
        "bitwise_or",
        "bitwise_right_shift",
        "bitwise_xor",
        "ceil",
        "clip",
        "conj",
        "copysign",
        "cos",
        "cosh",
        "divide",
        "equal",
        "exp",
        "expm1",
        "floor",
        "floor_divide",
        "greater",
        "greater_equal",
        "hypot",
        "imag",
        "isfinite",
        "isinf",
        "isnan",
        "less",
        "less_equal",
        "log",
        "log1p",
        "log2",
        "log10",
        "logaddexp",
        "logical_and",
        "logical_not",
        "logical_or",
        "logical_xor",
        "maximum",
        "minimum",
        "multiply",
        "negative",
        "nextafter",
        "not_equal",
        "positive",
        "pow",
        "real",
        "reciprocal",
        "remainder",
        "round",
        "sign",
        "signbit",
        "sin",
        "sinh",
        "square",
        "sqrt",
        "subtract",
        "tan",
        "tanh",
        "trunc",
        # Indexing Functions
        "take",
        "take_along_axis",
        # Linear Algebra Functions
        "matmul",
        "matrix_transpose",
        "tensordot",
        "vecdot",
        # Manipulation Functions
        "broadcast_arrays",
        "broadcast_to",
        "concat",
        "expand_dims",
        "flip",
        "moveaxis",
        "permute_dims",
        "repeat",
        "reshape",
        "roll",
        "squeeze",
        "stack",
        "tile",
        "unstack",
        # Searching Functions
        "argmax",
        "argmin",
        "count_nonzero",
        "nonzero",
        "searchsorted",
        "where",
        # Set functions
        "unique_all",
        "unique_counts",
        "unique_inverse",
        "unique_values",
        # Sorting Functions
        "argsort",
        "sort",
        # Statistical Functions
        "cumulative_prod",
        "cumulative_sum",
        "max",
        "mean",
        "min",
        "prod",
        "std",
        "sum",
        "var",
        # Utility Functions
        "all",
        "any",
        "diff",
    ],
    "fft": [
        "fft",
        "ifft",
        "fftn",
        "ifftn",
        "rfft",
        "irfft",
        "rfftn",
        "irfftn",
        "hfft",
        "ihfft",
        "fftfreq",
        "rfftfreq",
        "fftshift",
        "ifftshift",
    ],
    "linalg": [
        "cholesky",
        "cross",
        "det",
        "diagonal",
        "eigh",
        "eigvalsh",
        "inv",
        "matmul",
        "matrix_norm",
        "matrix_power",
        "matrix_rank",
        "matrix_transpose",
        "outer",
        "pinv",
        "qr",
        "slogdet",
        "solve",
        "svd",
        "svdvals",
        "tensordot",
        "trace",
        "vecdot",
        "vector_norm",
    ],
}

XFAILS = {
    ("numpy", ""): ["from_dlpack"] if np.__version__ < "1.23" else [],
    ("dask.array", ""): ["from_dlpack", "take_along_axis"],
    ("dask.array", "linalg"): [
        "cross",
        "det",
        "eigh",
        "eigvalsh",
        "matrix_power",
        "pinv",
        "slogdet",
    ],
}


@pytest.mark.parametrize("module", list(NAMES))
@pytest.mark.parametrize("library", wrapped_libraries)
def test_dir(library, module):
    """Test that dir() isn't missing any exports."""
    xp = pytest.importorskip(f"array_api_compat.{library}")
    mod = getattr(xp, module) if module else xp
    missing = set(NAMES[module]) - set(dir(mod))
    xfail = set(XFAILS.get((library, module), []))
    xpass = xfail - missing
    fails = missing - xfail
    assert not xpass, "Names in XFAILS are defined: %s" % xpass
    assert not fails, "Missing exports: %s" % fails


@pytest.mark.parametrize("module", list(NAMES))
@pytest.mark.parametrize("library", wrapped_libraries)
def test_all(library, module):
    """Test that __all__ isn't missing any exports."""
    modname = f"array_api_compat.{library}"
    pytest.importorskip(modname)
    if module:
        modname += f".{module}"

    objs = {}
    exec(f"from {modname} import *", objs)

    missing = set(NAMES[module]) - objs.keys()
    xfail = set(XFAILS.get((library, module), []))

    # FIXME are these canonically meant to be in __all__?
    if module == "":
        xfail |= {"__array_namespace_info__", "__array_api_version__"}

    xpass = xfail - missing
    fails = missing - xfail
    assert not xpass, "Names in XFAILS are defined: %s" % xpass
    assert not fails, "Missing exports: %s" % fails


@pytest.mark.parametrize(
    "name", [name for name in NAMES[""] if hasattr(builtins, name)]
)
@pytest.mark.parametrize("library", wrapped_libraries)
def test_builtins_collision(library, name):
    """Test that xp.bool is not accidentally builtins.bool, etc."""
    xp = pytest.importorskip(f"array_api_compat.{library}")
    assert getattr(xp, name) is not getattr(builtins, name)
