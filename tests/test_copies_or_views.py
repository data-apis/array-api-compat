"""
A collection of tests to make sure that wrapped namespaces agree with the bare ones
on whether to return a view or a copy of inputs.
"""
import pytest
from ._helpers import import_, wrapped_libraries


FUNC_INPUTS = [
    # func_name, arr_input, dtype,  scalar_value
    ('abs', [1, 2], 'int8', 3),
    ('abs', [1, 2], 'float32', 3.),
    ('ceil', [1, 2], 'int8', 3),
    ('clip', [1, 2], 'int8', 3),
    ('conj', [1, 2], 'int8', 3),
    ('floor', [1, 2], 'int8', 3),
    ('imag', [1j, 2j], 'complex64', 3),
    ('positive', [1, 2], 'int8', 3),
    ('real', [1., 2.], 'float32', 3.),
    ('round', [1, 2], 'int8', 3),
    ('sign', [0, 0], 'float32', 3),
    ('trunc', [1, 2], 'int8', 3),
    ('trunc', [1, 2], 'float32', 3),
]


def ensure_unary(func, arr):
    """Make a trivial unary function from func."""
    if func.__name__ == 'clip':
        return lambda x: func(x, arr[0], arr[1])
    return func


def is_view(func, a, value):
    """Apply `func`, mutate the output; does the input change?"""
    b = func(a)
    b[0] = value
    return a[0] == value


@pytest.mark.parametrize('xp_name', wrapped_libraries + ['array_api_strict'])
@pytest.mark.parametrize('inputs', FUNC_INPUTS, ids=[inp[0] for inp in FUNC_INPUTS])
def test_view_or_copy(inputs, xp_name):
    bare_xp = import_(xp_name, wrapper=False)
    wrapped_xp = import_(xp_name, wrapper=True)

    func_name, arr_input, dtype_str, value = inputs
    dtype = getattr(bare_xp, dtype_str)

    bare_func = getattr(bare_xp, func_name)
    bare_func = ensure_unary(bare_func, arr_input)

    wrapped_func = getattr(wrapped_xp, func_name)
    wrapped_func = ensure_unary(wrapped_func, arr_input)

    # bare namespace: mutate the output, does the input change?
    a = bare_xp.asarray(arr_input, dtype=dtype)
    is_view_bare = is_view(bare_func, a, value)

    # wrapped namespace: mutate the output, does the input change?
    a1 = wrapped_xp.asarray(arr_input, dtype=dtype)
    is_view_wrapped = is_view(wrapped_func, a1, value)

    assert is_view_bare == is_view_wrapped
