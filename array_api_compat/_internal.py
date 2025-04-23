"""
Internal helpers
"""

import importlib
from collections.abc import Callable
from functools import wraps
from inspect import signature
from types import ModuleType
from typing import TypeVar

_T = TypeVar("_T")


def get_xp(xp: ModuleType) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
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

    def inner(f: Callable[..., _T], /) -> Callable[..., _T]:
        @wraps(f)
        def wrapped_f(*args: object, **kwargs: object) -> object:
            return f(*args, xp=xp, **kwargs)

        sig = signature(f)
        new_sig = sig.replace(
            parameters=[par for i, par in sig.parameters.items() if i != "xp"]
        )

        if wrapped_f.__doc__ is None:
            wrapped_f.__doc__ = f"""\
Array API compatibility wrapper for {f.__name__}.

See the corresponding documentation in NumPy/CuPy and/or the array API
specification for more details.

"""
        wrapped_f.__signature__ = new_sig  # pyright: ignore[reportAttributeAccessIssue]
        return wrapped_f  # pyright: ignore[reportReturnType]

    return inner


def clone_module(mod_name: str, globals_: dict[str, object]) -> list[str]:
    """Import everything from module, updating globals().
    Returns __all__.
    """
    mod = importlib.import_module(mod_name)
    all_ = []
    for n in dir(mod):
        if not n.startswith("_") and hasattr(mod, n):
            all_.append(n)
            globals_[n] = getattr(mod, n)
    return all_


__all__ = ["get_xp", "clone_module"]

def __dir__() -> list[str]:
    return __all__
