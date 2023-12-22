from ..common import _aliases

from .._internal import get_xp

import numpy as np
from numpy import (
    # Constants
    e,
    inf,
    nan,
    pi,
    newaxis,
    # Dtypes
    bool_ as bool,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    complex64,
    complex128,
    iinfo,
    finfo,
    can_cast,
    result_type,
)

import dask.array as da

isdtype = get_xp(np)(_aliases.isdtype)
astype = _aliases.astype

# Common aliases
arange = get_xp(da)(_aliases.arange)
eye = get_xp(da)(_aliases.eye)

from functools import partial
asarray = partial(_aliases._asarray, namespace='dask')
asarray.__doc__ = _aliases._asarray.__doc__

linspace = get_xp(da)(_aliases.linspace)
eye = get_xp(da)(_aliases.eye)
UniqueAllResult = get_xp(da)(_aliases.UniqueAllResult)
UniqueCountsResult = get_xp(da)(_aliases.UniqueCountsResult)
UniqueInverseResult = get_xp(da)(_aliases.UniqueInverseResult)
unique_all = get_xp(da)(_aliases.unique_all)
unique_counts = get_xp(da)(_aliases.unique_counts)
unique_inverse = get_xp(da)(_aliases.unique_inverse)
unique_values = get_xp(da)(_aliases.unique_values)
permute_dims = get_xp(da)(_aliases.permute_dims)
std = get_xp(da)(_aliases.std)
var = get_xp(da)(_aliases.var)
empty = get_xp(da)(_aliases.empty)
empty_like = get_xp(da)(_aliases.empty_like)
full = get_xp(da)(_aliases.full)
full_like = get_xp(da)(_aliases.full_like)
ones = get_xp(da)(_aliases.ones)
ones_like = get_xp(da)(_aliases.ones_like)
zeros = get_xp(da)(_aliases.zeros)
zeros_like = get_xp(da)(_aliases.zeros_like)
reshape = get_xp(da)(_aliases.reshape)
matrix_transpose = get_xp(da)(_aliases.matrix_transpose)
vecdot = get_xp(da)(_aliases.vecdot)

from dask.array import (
    # Element wise aliases
    arccos as acos,
    arccosh as acosh,
    arcsin as asin,
    arcsinh as asinh,
    arctan as atan,
    arctan2 as atan2,
    arctanh as atanh,
    left_shift as bitwise_left_shift,
    right_shift as bitwise_right_shift,
    invert as bitwise_invert,
    power as pow,
    # Other
    concatenate as concat,
)

