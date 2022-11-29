from functools import partial

from ..common._aliases import *
from ..common._aliases import _asarray
from ..common._aliases import __all__

asarray = asarray_numpy = partial(_asarray, namespace='numpy')
asarray.__doc__ = _asarray.__doc__
del partial

__all__ = __all__ + ['asarray', 'asarray_numpy']
