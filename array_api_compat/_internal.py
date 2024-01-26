"""
Internal helpers
"""

from functools import wraps
from inspect import signature


def get_xp(xp):
    """
    Decorator to automatically replace xp with the corresponding array module.

    Use like

    import numpy as np

    @get_xp(np)
    def func(x, /, xp, kwarg=None):
        return xp.func(x, kwarg=kwarg)

    Note that xp must be a keyword argument and come after all non-keyword
    arguments.

    """
    def inner(f):
        @wraps(f)
        def wrapped_f(*args, **kwargs):
            return f(*args, xp=xp, **kwargs)

        sig = signature(f)
        new_sig = sig.replace(parameters=[sig.parameters[i] for i in sig.parameters if i != 'xp'])

        if wrapped_f.__doc__ is None:
            wrapped_f.__doc__ = f"""\
Array API compatibility wrapper for {f.__name__}.

See the corresponding documentation in NumPy/CuPy and/or the array API
specification for more details.

"""
        wrapped_f.__signature__ = new_sig
        return wrapped_f

    return inner


def _get_all_public_members(module, exclude=None, extend_all=False):
    """Get all public members of a module.
    
    Parameters
    ----------
    module : module
        The module to get members from.
    exclude : callable, optional
        A callable that takes a name and returns True if the name should be
        excluded from the list of members.
    extend_all : bool, optional
        If True, extend the module's __all__ attribute with the members of the
        module derive from dir(module)
    """
    members = getattr(module, '__all__', [])

    if members and not extend_all:
        return members

    if exclude is None:
        exclude = lambda name: name.startswith('_') # noqa: E731

    members += [_ for _  in dir(module) if not exclude(_)]

    # remove duplicates
    return list(set(members))