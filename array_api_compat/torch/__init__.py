from torch import *

# Several names are not included in the above import *
import torch
for n in dir(torch):
    if not n.startswith('_'):
        exec(n + ' = torch.' + n)

# These imports may overwrite names from the import * above.
from ._aliases import *

from ..common._helpers import *

__array_api_version__ = '2021.12'
