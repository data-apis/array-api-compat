from array_api_compat import (is_numpy_array, is_cupy_array, is_torch_array,
                              is_dask_array, is_array_api_obj)

from ._helpers import import_

import pytest

is_functions = {
    'numpy': 'is_numpy_array',
    'cupy': 'is_cupy_array',
    'torch': 'is_torch_array',
    'dask.array': 'is_dask_array',
}

@pytest.mark.parametrize('library', is_functions.keys())
@pytest.mark.parametrize('func', is_functions.values())
def test_is_xp_array(library, func):
    lib = import_(library)
    is_func = globals()[func]

    x = lib.asarray([1, 2, 3])

    assert is_func(x) == (func == is_functions[library])

    assert is_array_api_obj(x)
