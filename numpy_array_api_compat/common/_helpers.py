"""
Various helper functions which are not part of the spec.
"""
def get_namespace(*xs, _use_compat=True):
    """
    Get the array API compatible namespace for the arrays `xs`.

    `xs` should contain one or more arrays.
    """
    from ..numpy._helpers import _is_numpy_array

    namespaces = set()
    for x in xs:
        if isinstance(x, (tuple, list)):
            namespaces.add(get_namespace(*x, _use_compat=_use_compat))
        elif hasattr(x, '__array_namespace__'):
            namespaces.add(x.__array_namespace__)
        elif _is_numpy_array(x):
            if _use_compat:
                from .. import numpy as numpy_namespace
                namespaces.add(numpy_namespace)
            else:
                import numpy as np
                namespaces.add(np)
        else:
            # TODO: Support Python scalars?
            raise ValueError("The input is not a supported array type")

    if not namespaces:
        raise ValueError("Unrecognized array input")

    if len(namespaces) != 1:
        raise ValueError(f"Multiple namespaces for array inputs: {namespaces}")

    xp, = namespaces

    return xp
