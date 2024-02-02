import torch as _torch

from .._internal import _get_all_public_members

_torch_linalg_all = _get_all_public_members(_torch.linalg)

for _name in _torch_linalg_all:
    globals()[_name] = getattr(_torch.linalg, _name)

# outer is implemented in torch but aren't in the linalg namespace
outer = _torch.outer

from ._aliases import (  # noqa: E402
    matrix_transpose,
    solve,
    sum,
    tensordot,
    trace,
    vecdot_linalg as vecdot,
)

__all__ = []

__all__ += _torch_linalg_all

__all__ += [
    "matrix_transpose",
    "outer",
    "solve",
    "sum",
    "tensordot",
    "trace",
    "vecdot",
]
