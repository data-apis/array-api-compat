# ruff: noqa: PLC0414
from typing import Final

from numpy import *  # noqa: F403  # pyright: ignore[reportWildcardImportFromLibrary]

# from numpy import * doesn't overwrite these builtin names
from numpy import abs as abs
from numpy import max as max
from numpy import min as min
from numpy import round as round

# These imports may overwrite names from the import * above.
from ._aliases import *  # noqa: F403

# Don't know why, but we have to do an absolute import to import linalg. If we
# instead do
#
# from . import linalg
#
# It doesn't overwrite np.linalg from above. The import is generated
# dynamically so that the library can be vendored.
__import__(__package__ + ".linalg")

__import__(__package__ + ".fft")

from ..common._helpers import *  # noqa: F403
from .linalg import matrix_transpose, vecdot  # noqa: F401

try:
    # Used in asarray(). Not present in older versions.
    from numpy import _CopyMode  # noqa: F401
except ImportError:
    pass

__array_api_version__: Final = "2024.12"
