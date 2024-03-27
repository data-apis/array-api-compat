"""
isdtype is not yet tested in the test suite, and it should extend properly to
non-spec dtypes
"""

import pytest

from ._helpers import import_, wrapped_libraries

# Check the known dtypes by their string names

def _spec_dtypes(library):
    if library == 'torch':
        # torch does not have unsigned integer dtypes
        return {
            'bool',
            'complex64',
            'complex128',
            'uint8',
            'int8',
            'int16',
            'int32',
            'int64',
            'float32',
            'float64',
        }
    else:
        return {
            'bool',
            'complex64',
            'complex128',
            'float32',
            'float64',
            'int16',
            'int32',
            'int64',
            'int8',
            'uint16',
            'uint32',
            'uint64',
            'uint8',
        }

dtype_categories = {
    'bool': lambda d: d == 'bool',
    'signed integer': lambda d: d.startswith('int'),
    'unsigned integer': lambda d: d.startswith('uint'),
    'integral': lambda d: dtype_categories['signed integer'](d) or
                          dtype_categories['unsigned integer'](d),
    'real floating': lambda d: 'float' in d,
    'complex floating': lambda d: d.startswith('complex'),
    'numeric': lambda d: dtype_categories['integral'](d) or
                         dtype_categories['real floating'](d) or
                         dtype_categories['complex floating'](d),
}

def isdtype_(dtype_, kind):
    # Check a dtype_ string against kind. Note that 'bool' technically has two
    # meanings here but they are both the same.
    if kind in dtype_categories:
        res = dtype_categories[kind](dtype_)
    else:
        res = dtype_ == kind
    assert type(res) is bool #  noqa: E721
    return res

@pytest.mark.parametrize("library", wrapped_libraries)
def test_isdtype_spec_dtypes(library):
    xp = import_(library, wrapper=True)

    isdtype = xp.isdtype

    for dtype_ in _spec_dtypes(library):
        for dtype2_ in _spec_dtypes(library):
            dtype = getattr(xp, dtype_)
            dtype2 = getattr(xp, dtype2_)
            res = isdtype_(dtype_, dtype2_)
            assert isdtype(dtype, dtype2) is res, (dtype_, dtype2_)

        for cat in dtype_categories:
            res = isdtype_(dtype_, cat)
            assert isdtype(dtype, cat) == res, (dtype_, cat)

        # Basic tuple testing (the array-api testsuite will be more complete here)
        for kind1_ in [*_spec_dtypes(library), *dtype_categories]:
            for kind2_ in [*_spec_dtypes(library), *dtype_categories]:
                kind1 = kind1_ if kind1_ in dtype_categories else getattr(xp, kind1_)
                kind2 = kind2_ if kind2_ in dtype_categories else getattr(xp, kind2_)
                kind = (kind1, kind2)

                res = isdtype_(dtype_, kind1_) or isdtype_(dtype_, kind2_)
                assert isdtype(dtype, kind) == res, (dtype_, (kind1_, kind2_))

additional_dtypes = [
    'float16',
    'float128',
    'complex256',
    'bfloat16',
]

@pytest.mark.parametrize("library", wrapped_libraries)
@pytest.mark.parametrize("dtype_", additional_dtypes)
def test_isdtype_additional_dtypes(library, dtype_):
    xp = import_(library, wrapper=True)

    isdtype = xp.isdtype

    if not hasattr(xp, dtype_):
        return
        # pytest.skip(f"{library} doesn't have dtype {dtype_}")

    dtype = getattr(xp, dtype_)
    for cat in dtype_categories:
        res = isdtype_(dtype_, cat)
        assert isdtype(dtype, cat) == res, (dtype_, cat)
