import functools
import sys
import types

from ...common import is_numpy_namespace

__all__ = ['wrap_namespace']
_all_ignore = ['functools', 'sys', 'types', 'is_numpy_namespace']


def wrap_namespace(xp):
    """Create a bespoke Dask namespace that wraps around another namespace.

    Parameters
    ----------
    xp : namespace
        Namespace to be wrapped by Dask

    Returns
    -------
    namespace :
        A module object that duplicates array_api_compat.dask.array, with the
        difference that all creation functions will create an array with the same
        meta namespace as the input.
    """
    from .. import array as da_compat

    if is_numpy_namespace(xp):
        return da_compat

    mod_name = f'{da_compat.__name__}.{xp.__name__}'
    try:
        return sys.modules[mod_name]
    except KeyError:
        pass

    mod = types.ModuleType(mod_name)
    sys.modules[mod_name] = mod

    meta = xp.empty(())
    for name, v in da_compat.__dict__.items():
        if name.startswith('_'):
            continue
        if name in {'arange',  'asarray', 'empty', 'eye', 'from_dlpack',
                    'full', 'linspace', 'ones', 'zeros'}:
            v = functools.wraps(v)(functools.partial(v, like=meta))
        setattr(mod, name, v)

    return mod
