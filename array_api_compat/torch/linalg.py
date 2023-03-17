from torch.linalg import *

# torch.linalg doesn't define __all__
# from torch.linalg import __all__ as linalg_all
from torch import linalg as _linalg
linalg_all = [i for i in dir(_linalg) if not i.startswith('_')]

# These are implemented in torch but aren't in the linalg namespace
from torch import outer, trace
from ._aliases import matrix_transpose, tensordot

__all__ = linalg_all + ['outer', 'trace', 'matrix_transpose', 'tensordot']

del linalg_all
del _linalg
