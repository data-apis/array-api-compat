from __future__ import annotations

from functools import partial

from ..common import _aliases

from .._internal import get_xp

asarray = asarray_numpy = partial(_aliases._asarray, namespace='numpy')
asarray.__doc__ = _aliases._asarray.__doc__
del partial

import numpy as np
bool = np.bool_

acos = get_xp(np)(_aliases.acos)
acosh = get_xp(np)(_aliases.acosh)
asin = get_xp(np)(_aliases.asin)
asinh = get_xp(np)(_aliases.asinh)
atan = get_xp(np)(_aliases.atan)
atan2 = get_xp(np)(_aliases.atan2)
atanh = get_xp(np)(_aliases.atanh)
bitwise_left_shift = get_xp(np)(_aliases.bitwise_left_shift)
bitwise_invert = get_xp(np)(_aliases.bitwise_invert)
bitwise_right_shift = get_xp(np)(_aliases.bitwise_right_shift)
concat = get_xp(np)(_aliases.concat)
pow = get_xp(np)(_aliases.pow)
arange = get_xp(np)(_aliases.arange)
empty = get_xp(np)(_aliases.empty)
empty_like = get_xp(np)(_aliases.empty_like)
eye = get_xp(np)(_aliases.eye)
full = get_xp(np)(_aliases.full)
full_like = get_xp(np)(_aliases.full_like)
linspace = get_xp(np)(_aliases.linspace)
ones = get_xp(np)(_aliases.ones)
ones_like = get_xp(np)(_aliases.ones_like)
zeros = get_xp(np)(_aliases.zeros)
zeros_like = get_xp(np)(_aliases.zeros_like)
UniqueAllResult = get_xp(np)(_aliases.UniqueAllResult)
UniqueCountsResult = get_xp(np)(_aliases.UniqueCountsResult)
UniqueInverseResult = get_xp(np)(_aliases.UniqueInverseResult)
unique_all = get_xp(np)(_aliases.unique_all)
unique_counts = get_xp(np)(_aliases.unique_counts)
unique_inverse = get_xp(np)(_aliases.unique_inverse)
unique_values = get_xp(np)(_aliases.unique_values)
astype = _aliases.astype
std = get_xp(np)(_aliases.std)
var = get_xp(np)(_aliases.var)
permute_dims = get_xp(np)(_aliases.permute_dims)
reshape = get_xp(np)(_aliases.reshape)
argsort = get_xp(np)(_aliases.argsort)
sort = get_xp(np)(_aliases.sort)
sum = get_xp(np)(_aliases.sum)
prod = get_xp(np)(_aliases.prod)
ceil = get_xp(np)(_aliases.ceil)
floor = get_xp(np)(_aliases.floor)
trunc = get_xp(np)(_aliases.trunc)

__all__ = _aliases.__all__ + ['asarray', 'asarray_numpy', 'bool', 'arange',
                              'empty', 'empty_like', 'eye', 'full', 'full_like',
                              'linspace', 'ones', 'ones_like', 'zeros', 'zeros_like']
