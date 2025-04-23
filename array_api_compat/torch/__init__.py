from typing import Final

from torch import * # noqa: F403

# Several names are not included in the above import *
_torch_all = set()
import torch
for n in dir(torch):
    if (n.startswith('_')
        or n.endswith('_')
        or 'cuda' in n
        or 'cpu' in n
        or 'backward' in n):
        continue
    exec(f"{n} = torch.{n}")
    _torch_all.add(n)
del n

# These imports may overwrite names from the import * above.
import _aliases
from ._aliases import * # noqa: F403
from ._info import __array_namespace_info__  # noqa: F401

# See the comment in the numpy __init__.py
__import__(__package__ + '.linalg')
__import__(__package__ + '.fft')

__array_api_version__: Final = '2024.12'

__all__ = sorted(
    set(_torch_all)
    | set(_aliases.__all__)
    | {"__array_api_version__", "__array_namespace_info__", "linalg", "fft"}
)

def __dir__() -> list[str]:
    return __all__
