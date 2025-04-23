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


def all_names(mod):
    """Return all names imported by `from mod import *`.
    This is typically `__all__` but, if not defined, Python
    implements automated fallbacks.
    """
    objs = {}
    exec(f"from {mod.__name__} import *", objs)
    return list(objs)


@pytest.mark.parametrize("func", [all_names, dir])
@pytest.mark.parametrize("module", list(NAMES))
@pytest.mark.parametrize("library", wrapped_libraries)
def test_array_api_names(library, module, func):
    """Test that __all__ and dir() aren't missing any exports
    dictated by the Standard.
    """
    xp = pytest.importorskip(f"array_api_compat.{library}")
    mod = getattr(xp, module) if module else xp
    missing = set(NAMES[module]) - set(func(mod))
    xfail = set(XFAILS.get((library, module), []))
    xpass = xfail - missing
    fails = missing - xfail
    assert not xpass, f"Names in XFAILS are defined: {xpass}"
    assert not fails, f"Missing exports: {fails}"


@pytest.mark.parametrize("func", [all_names, dir])
@pytest.mark.parametrize("module", list(NAMES))
@pytest.mark.parametrize("library", wrapped_libraries)
def test_compat_doesnt_hide_names(library, module, func):
    """The base namespace can have more names than the ones explicitly exported
    by array-api-compat. Test that we're not suppressing them.
    """
    bare_xp = pytest.importorskip(library)
    compat_xp = pytest.importorskip(f"array_api_compat.{library}")
    bare_mod = getattr(bare_xp, module) if module else bare_xp
    compat_mod = getattr(compat_xp, module) if module else compat_xp

    missing = set(func(bare_mod)) - set(func(compat_mod))
    missing = {name for name in missing if not name.startswith("_")}
    assert not missing, f"Non-Array API names have been hidden: {missing}"


@pytest.mark.parametrize("func", [all_names, dir])
@pytest.mark.parametrize("module", list(NAMES))
@pytest.mark.parametrize("library", wrapped_libraries)
def test_compat_doesnt_add_names(library, module, func):
    """Test that array-api-compat isn't adding names to the namespace
    besides those defined by the Array API Standard.
    """
    bare_xp = pytest.importorskip(library)
    compat_xp = pytest.importorskip(f"array_api_compat.{library}")
    bare_mod = getattr(bare_xp, module) if module else bare_xp
    compat_mod = getattr(compat_xp, module) if module else compat_xp

    aapi_names = set(NAMES[module])
    spurious = set(func(compat_mod)) - set(func(bare_mod)) - aapi_names - {"__all__"}
    # Quietly ignore *Result dataclasses
    spurious = {name for name in spurious if not name.endswith("Result")}
    assert not spurious, (
        f"array-api-compat is adding non-Array API names: {spurious}"
    )


@pytest.mark.parametrize(
    "name", [name for name in NAMES[""] if hasattr(builtins, name)]
)
@pytest.mark.parametrize("library", wrapped_libraries)
def test_builtins_collision(library, name):
    """Test that xp.bool is not accidentally builtins.bool, etc."""
    xp = pytest.importorskip(f"array_api_compat.{library}")
    assert getattr(xp, name) is not getattr(builtins, name)
