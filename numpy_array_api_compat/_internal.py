"""
Internal helpers
"""

from functools import wraps
from inspect import signature

from ._helpers import get_namespace

def get_xp(f):
    """
    Decorator to automatically replace xp with the corresponding array module

    Use like

    @get_xp
    def func(x, /, xp, kwarg=None):
        return xp.func(x, kwarg=kwarg)

    Note that xp must be able to be passed as a keyword argument.
    """
    @wraps(f)
    def inner(*args, **kwargs):
        xp = get_namespace(*args, _use_compat=False)
        return f(*args, xp=xp, **kwargs)

    sig = signature(f)
    new_sig = sig.replace(parameters=[sig.parameters[i] for i in sig.parameters if i != 'xp'])

    inner.__signature__ = new_sig

    return inner
