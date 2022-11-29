from cupy.linalg import *
# cupy.linalg doesn't have __all__. If it is added, replace this with
#
# from cupy.linalg import __all__ as linalg_all
_n = {}
exec('from cupy.linalg import *', _n)
del _n['__builtins__']
linalg_all = list(_n)
del _n

from ..common.linalg import *
from ..common.linalg import __all__ as common_linalg_all

__all__ = linalg_all + common_linalg_all
